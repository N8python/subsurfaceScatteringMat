import * as THREE from 'https://cdn.skypack.dev/three@0.150.0';
import { FullScreenQuad } from "https://unpkg.com/three@0.150.0/examples/jsm/postprocessing/Pass.js";
const translucentHBlur = {
    uniforms: {
        tDiffuse: { value: null },
        tDepth: { value: null },
        resolution: { value: new THREE.Vector2() },
        size: { value: 4.0 },
        stride: { value: 8.0 },
        near: { value: 0.1 },
        far: { value: 1000.0 },
        orthographic: { value: false },
        logDepthBuffer: { value: false }
    },
    vertexShader: /*glsl*/ `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
        `,
    fragmentShader: /*glsl*/ `
        uniform highp sampler2D tDiffuse;
        uniform highp sampler2D tDepth;
        uniform float size;
        uniform float stride;
        uniform float near;
        uniform float far;
        uniform bool orthographic;
        uniform bool logDepthBuffer;
        uniform vec2 resolution;
        varying vec2 vUv;
        highp float linearize_depth(highp float d, highp float zNear,highp float zFar)
        {
            highp float z_n = 2.0 * d - 1.0;
            return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
        }
          highp float linearize_depth_log(highp float d, highp float nearZ,highp float farZ) {
        float depth = pow(2.0, d * log2(farZ + 1.0)) - 1.0;
        float a = farZ / (farZ - nearZ);
        float b = farZ * nearZ / (nearZ - farZ);
        float linDepth = a + b / depth;
        return linearize_depth(linDepth, nearZ, farZ);
      }
        void main() {
            vec2 uv = vUv;
            float depth = orthographic ? near + (far - near) * texture2D(tDepth, uv).x:
            (
                logDepthBuffer ?
                linearize_depth_log(texture2D(tDepth, uv).x, near, far) :
                linearize_depth(texture2D(tDepth, uv).x, near, far)
            );              // Attenuate the stride based on the depth, so that the blur is larger near the camera
            float updatedStride = stride * (
                1.0 / (1.0 + (depth - near))
                );
            vec2 invResolution = 1.0 / resolution;
            vec3 color = vec3(0.0);
            float total = 0.0;
             if (texture2D(tDiffuse, uv).a == 0.0) {
                gl_FragColor = vec4(0.0);
                return;
            }
            // Calculate the average of the surrounding pixels, using a horizontal Gaussian filter
            // We take advantage of the fact that the Gaussian filter is separable, so we can do two passes
            // Weight each pixel by the Gaussian function and its alpha value
            for (float x = -size; x <= size; x += 1.0) {
                vec4 texel = texture2D(tDiffuse, uv + vec2(x, 0.0)* updatedStride * invResolution);
                float weight = exp(-0.5 * (x * x) / (size * size)) * texel.a;
                color += max(texel.rgb, vec3(0.0)) * weight;
                total += weight;
            }
            color /= total;
            gl_FragColor = vec4(color, 1.0);
        }`
};
const translucentVBlur = {
    uniforms: {
        tDiffuse: { value: null },
        tDepth: { value: null },
        resolution: { value: new THREE.Vector2() },
        size: { value: 4.0 },
        stride: { value: 8.0 },
        near: { value: 0.1 },
        far: { value: 1000.0 },
        orthographic: { value: false },
        logDepthBuffer: { value: false }
    },
    vertexShader: /*glsl*/ `
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
    `,
    fragmentShader: /*glsl*/ `
    uniform highp sampler2D tDiffuse;
    uniform highp sampler2D tDepth;
    uniform float size;
    uniform float stride;
    uniform float near;
    uniform float far;
    uniform bool orthographic;
    uniform bool logDepthBuffer;
    uniform vec2 resolution;
    varying vec2 vUv;
    highp float linearize_depth(highp float d, highp float zNear,highp float zFar)
    {
        highp float z_n = 2.0 * d - 1.0;
        return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
    }
    highp float linearize_depth_log(highp float d, highp float nearZ,highp float farZ) {
        float depth = pow(2.0, d * log2(farZ + 1.0)) - 1.0;
        float a = farZ / (farZ - nearZ);
        float b = farZ * nearZ / (nearZ - farZ);
        float linDepth = a + b / depth;
        return linearize_depth(linDepth, nearZ, farZ);
      }
    void main() {
        vec2 uv = vUv;
            float depth = orthographic ? near + (far - near) * texture2D(tDepth, uv).x:
            (
                logDepthBuffer ?
                linearize_depth_log(texture2D(tDepth, uv).x, near, far) :
                linearize_depth(texture2D(tDepth, uv).x, near, far)
            );        
            float updatedStride = stride * (
            1.0 / (1.0 + (depth - near))
            );
        vec2 invResolution = 1.0 / resolution;
        vec3 color = vec3(0.0);
        float total = 0.0;
        if (texture2D(tDiffuse, uv).a == 0.0) {
            gl_FragColor = vec4(0.0);
            return;
        }
        // Calculate the average of the surrounding pixels, using a horizontal Gaussian filter
        // We take advantage of the fact that the Gaussian filter is separable, so we can do two passes
        // Weight each pixel by the Gaussian function and its alpha value
        for (float y = -size; y <= size; y += 1.0) {
            vec4 texel = texture2D(tDiffuse, uv + vec2(0.0, y) * updatedStride * invResolution);
            float weight = exp(-0.5 * (y * y) / (size * size)) * texel.a;
            color += max(texel.rgb, vec3(0.0)) * weight;
            total += weight;
        }
        color /= total;
        gl_FragColor = vec4(color, 1.0);

    }`
};
const depthBlit = {
    uniforms: {
        tDiffuse: { value: null },
    },
    vertexShader: /*glsl*/ `
    varying vec2 vUv;
    void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
    `,
    fragmentShader: /*glsl*/ `
    varying vec2 vUv;
    uniform highp sampler2D tDiffuse;
    highp float linearize_depth(highp float d, highp float zNear,highp float zFar)
    {
        highp float z_n = 2.0 * d - 1.0;
        return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
    }
    void main() {
        gl_FragColor = vec4(texture2D(tDiffuse, vUv).x, 0.0, 0.0, 1.0);
    }
    `
}
const translucentHBlurQuad = new FullScreenQuad(new THREE.ShaderMaterial(translucentHBlur));
const translucentVBlurQuad = new FullScreenQuad(new THREE.ShaderMaterial(translucentVBlur));
const depthBlitQuad = new FullScreenQuad(new THREE.ShaderMaterial(depthBlit));
const thicknessMaterial = new THREE.ShaderMaterial({
    depthWrite: false,
    depthTest: false,
    transparent: true,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    uniforms: {},
    vertexShader: /*glsl*/ `
          varying vec3 vWorldPosition;
          void main() {
              vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
          `,
    fragmentShader: /*glsl*/ `
          varying vec3 vWorldPosition;
          void main() {
              gl_FragColor = vec4(vec3(distance(cameraPosition, vWorldPosition) * (gl_FrontFacing ? -1.0 : 1.0)), 1.0);
          }
          `
});
thicknessMaterial.forceSinglePass = true;

const depthWriteOnlyMaterial = new THREE.MeshBasicMaterial({
    colorWrite: false,
    depthWrite: true,
    side: THREE.DoubleSide,
});

class MeshTranslucentMaterial extends THREE.MeshPhysicalMaterial {
    constructor(parameters) {
        super();
        this.isMeshTranslucentMaterial = true;
        this.thicknessRenderTarget = null;
        this.thicknessRenderTargetBlur = null;
        this.thicknessRenderTargetDepth = null;
        this.renderTargetSize = new THREE.Vector2();
        this.roughnessBlurScale = 16.0;
        this.resolutionScale = 0.5;
        this.scattering = 1.0;
        this.scatteringAbsorption = 1.0;
        this.internalRoughness = 0.5;
        this.setValues(parameters);
        /*this.onBeforeRender = (renderer, scene, camera, geometry, material, group) => {
          material.uniforms.uCameraPosition.value.copy(camera.position);
        };*/
    }
    onBeforeRender(renderer, scene, camera, geometry, object, group) {
        renderer.getDrawingBufferSize(this.renderTargetSize);
        this.renderTargetSize.x = Math.floor(this.renderTargetSize.x * this.resolutionScale);
        this.renderTargetSize.y = Math.floor(this.renderTargetSize.y * this.resolutionScale);
        this.updateRenderTargets();
        this.updateInternalUniforms();
        const originClearAlpha = renderer.getClearAlpha();
        const originAutoClear = renderer.autoClear;
        const originRenderTarget = renderer.getRenderTarget();
        renderer.setRenderTarget(this.thicknessRenderTarget);
        renderer.setClearAlpha(0.0);
        renderer.clear();
        renderer.autoClear = false;
        object.material = thicknessMaterial;
        renderer.render(object, camera);
        object.material = depthWriteOnlyMaterial;
        renderer.render(object, camera);
        renderer.setClearAlpha(originClearAlpha);
        object.material = this;
        renderer.autoClear = originAutoClear;
        renderer.setRenderTarget(this.thicknessRenderTargetDepth);
        renderer.clear();
        depthBlitQuad.material.uniforms.tDiffuse.value = this.thicknessRenderTarget.depthTexture;
        depthBlitQuad.render(renderer);
        translucentHBlurQuad.material.uniforms.tDiffuse.value = this.thicknessRenderTarget.texture;
        translucentHBlurQuad.material.uniforms.tDepth.value = this.thicknessRenderTargetDepth.texture;
        translucentHBlurQuad.material.uniforms.resolution.value.copy(this.renderTargetSize);
        translucentHBlurQuad.material.uniforms.stride.value = this.roughnessBlurScale * 8.0 * this.roughness;
        translucentHBlurQuad.material.uniforms.near.value = camera.near;
        translucentHBlurQuad.material.uniforms.far.value = camera.far;
        translucentHBlurQuad.material.uniforms.orthographic.value = camera.isOrthographicCamera;
        translucentHBlurQuad.material.uniforms.logDepthBuffer.value = renderer.capabilities.logarithmicDepthBuffer;
        translucentVBlurQuad.material.uniforms.tDiffuse.value = this.thicknessRenderTargetBlur.texture;
        translucentVBlurQuad.material.uniforms.tDepth.value = this.thicknessRenderTargetDepth.texture;
        translucentVBlurQuad.material.uniforms.resolution.value.copy(this.renderTargetSize);
        translucentVBlurQuad.material.uniforms.stride.value = this.roughnessBlurScale * 8.0 * this.roughness;
        translucentVBlurQuad.material.uniforms.near.value = camera.near;
        translucentVBlurQuad.material.uniforms.far.value = camera.far;
        translucentVBlurQuad.material.uniforms.orthographic.value = camera.isOrthographicCamera;
        translucentVBlurQuad.material.uniforms.logDepthBuffer.value = renderer.capabilities.logarithmicDepthBuffer;
        renderer.setRenderTarget(this.thicknessRenderTargetBlur);
        translucentHBlurQuad.render(renderer);
        renderer.setRenderTarget(this.thicknessRenderTarget);
        translucentVBlurQuad.render(renderer);
        translucentHBlurQuad.material.uniforms.stride.value = this.roughnessBlurScale * 1.0 * this.roughness;
        translucentVBlurQuad.material.uniforms.stride.value = this.roughnessBlurScale * 1.0 * this.roughness;
        renderer.setRenderTarget(this.thicknessRenderTargetBlur);
        translucentHBlurQuad.render(renderer);
        renderer.setRenderTarget(this.thicknessRenderTarget);
        translucentVBlurQuad.render(renderer);
        renderer.setRenderTarget(originRenderTarget);
    }
    updateInternalUniforms() {
        if (!this._internalShader) return;
        if (this.thicknessRenderTarget) {
            this._internalShader.uniforms.thicknessTexture.value = this.thicknessRenderTarget.texture;
        }
        this._internalShader.uniforms.scattering.value = this.scattering;
        this._internalShader.uniforms.internalRoughness.value = this.internalRoughness;
        this._internalShader.uniforms.scatteringAbsorption.value = this.scatteringAbsorption;
    }
    onBeforeCompile(shader) {
        this._internalShader = shader;
        shader.uniforms.thicknessTexture = { value: this.thicknessRenderTarget.texture };
        shader.uniforms.scattering = { value: this.scattering };
        shader.uniforms.internalRoughness = { value: this.internalRoughness };
        shader.uniforms.scatteringAbsorption = { value: this.scatteringAbsorption };
        shader.fragmentShader = "uniform sampler2D thicknessTexture;\nuniform mat4 projectionMatrix;\nuniform float attenuationDistance;\nuniform float scattering;\nuniform float internalRoughness;\nuniform float scatteringAbsorption;\n" + shader.fragmentShader.replace(
            "#include <transmission_pars_fragment>",
            /*glsl*/
            `
                uniform float transmission;
                uniform float thickness;
                uniform vec3 attenuationColor;
                #ifdef USE_TRANSMISSIONMAP
                uniform sampler2D transmissionMap;
                #endif
                #ifdef USE_THICKNESSMAP
                uniform sampler2D thicknessMap;
                #endif
                uniform vec2 transmissionSamplerSize;
                uniform sampler2D transmissionSamplerMap;
                uniform mat4 modelMatrix;
                varying vec3 vWorldPosition;
                vec3 getVolumeTransmissionRay(const in vec3 n,
                    const in vec3 v,
                    const in float thickness,
                    const in float ior,
                    const in mat4 modelMatrix) {
                    // Direction of refracted light.
                    vec3 refractionVector = refract(-v, normalize(n), 1.0 / ior);
                    // Compute rotation-independant scaling of the model matrix.
                    vec3 modelScale;
                    modelScale.x = length(vec3(modelMatrix[0].xyz));
                    modelScale.y = length(vec3(modelMatrix[1].xyz));
                    modelScale.z = length(vec3(modelMatrix[2].xyz));
                    // The thickness is specified in local space.
                    return normalize(refractionVector) * thickness * modelScale;
                }
                float applyIorToRoughness(const in float roughness,
                    const in float ior) {
                    // Scale roughness with IOR so that an IOR of 1.0 results in no microfacet refraction and
                    // an IOR of 1.5 results in the default amount of microfacet refraction.
                    return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
                }
                vec4 getTransmissionSample(const in vec2 fragCoord,
                    const in float roughness,
                    const in float ior) {
                    float lod = log2(transmissionSamplerSize.x) * applyIorToRoughness(roughness, ior);
                    return textureBicubic(transmissionSamplerMap, fragCoord.xy, lod);
                }
                vec3 applyVolumeAttenuation(const in vec3 radiance,
                    const in float transmissionDistance,
                    const in vec3 attenuationColor,
                    const in float attenuationDistance) {
                    if (isinf(attenuationDistance)) {
                        // Attenuation distance is +∞, i.e. the transmitted color is not attenuated at all.
                        return radiance;
                    } else {
                        // Compute light attenuation using Beer's law.
                        vec3 attenuationCoefficient = -log(attenuationColor) / attenuationDistance;
                        vec3 transmittance = exp(-attenuationCoefficient * transmissionDistance); // Beer's law
                        return transmittance * radiance;
                    }
                }
                vec4 getIBLVolumeRefraction(const in vec3 n,
                    const in vec3 v,
                    const in float roughness,
                    const in vec3 diffuseColor,
                    const in vec3 specularColor,
                    const in float specularF90,
                    const in vec3 position,
                    const in mat4 modelMatrix,
                    const in mat4 viewMatrix,
                    const in mat4 projMatrix,
                    const in float ior,
                    const in float thickness,
                    const in vec3 attenuationColor,
                    const in float attenuationDistance) {
                    vec4 thicknessPos = projMatrix * viewMatrix * vec4(vWorldPosition, 1.0);
                    vec2 thicknessCoords = thicknessPos.xy / thicknessPos.w;
                    thicknessCoords += 1.0;
                    thicknessCoords /= 2.0;    
                    float viewRayDepth = texture2D(thicknessTexture, thicknessCoords).x;
                    vec3 transmissionRay = getVolumeTransmissionRay(n, v, viewRayDepth + thickness, ior, modelMatrix);
                    vec3 refractedRayExit = position + transmissionRay;
                    // Project refracted vector on the framebuffer, while mapping to normalized device coordinates.
                    vec4 ndcPos = projMatrix * viewMatrix * vec4(refractedRayExit, 1.0);
                    vec2 refractionCoords = ndcPos.xy / ndcPos.w;
                    refractionCoords += 1.0;
                    refractionCoords /= 2.0;
                    // Sample framebuffer to get pixel the refracted ray hits.
                    vec4 transmittedLight = getTransmissionSample(refractionCoords, roughness, ior);
                    vec3 attenuatedColor = applyVolumeAttenuation(transmittedLight.rgb, viewRayDepth, attenuationColor, attenuationDistance);
                    // Get the specular component.
                    vec3 F = EnvironmentBRDF(n, v, specularColor, specularF90, roughness);
                    return vec4((1.0 - F) * attenuatedColor * diffuseColor, transmittedLight.a);
                }
                `
        ).replace("#include <lights_physical_pars_fragment>",
            `
struct PhysicalMaterial {
	vec3 diffuseColor;
	float roughness;
	vec3 specularColor;
	float specularF90;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
};
// temporary
vec3 clearcoatSpecular = vec3( 0.0 );
vec3 sheenSpecular = vec3( 0.0 );
// This is a curve-fit approxmation to the "Charlie sheen" BRDF integrated over the hemisphere from 
// Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF". The analysis can be found
// in the Sheen section of https://drive.google.com/file/d/1T0D1VSyR4AllqIJTQAraEIzjlb5h4FKH/view?usp=sharing
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9 : -8.48 * r2 + 14.3 * roughness - 9.95;
	float b = roughness < 0.25 ? 44.0 * r2 - 23.7 * roughness + 3.26 : 1.97 * r2 - 3.27 * roughness + 0.72;
	float DG = exp( a * dotNV + b ) + ( roughness < 0.25 ? 0.0 : 0.1 * ( roughness - 0.25 ) );
	return saturate( DG * RECIPROCAL_PI );
}
// Analytical approximation of the DFG LUT, one half of the
// split-sum approximation used in indirect specular lighting.
// via 'environmentBRDF' from "Physically Based Shading on Mobile"
// https://www.unrealengine.com/blog/physically-based-shading-on-mobile
vec2 DFGApprox( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	const vec4 c0 = vec4( - 1, - 0.0275, - 0.572, 0.022 );
	const vec4 c1 = vec4( 1, 0.0425, 1.04, - 0.04 );
	vec4 r = roughness * c0 + c1;
	float a004 = min( r.x * r.x, exp2( - 9.28 * dotNV ) ) * r.x + r.y;
	vec2 fab = vec2( - 1.04, 1.04 ) * a004 + r.zw;
	return fab;
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	return specularColor * fab.x + specularF90 * fab.y;
}
// Fdez-Agüera's "Multiple-Scattering Microfacet Model for Real-Time Image Based Lighting"
// Approximates multiscattering in order to preserve energy.
// http://www.jcgt.org/published/0008/01/03/
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	vec2 fab = DFGApprox( normal, viewDir, roughness );
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619; // 1/21
	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometry.normal;
		vec3 viewDir = geometry.viewDir;
		vec3 position = geometry.position;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight; // counterclockwise; light shines in local neg z direction
		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		// LTC Fresnel Approximation by Stephen Hill
		// http://blog.selfshadow.com/publications/s2016-advances/s2016_ltc_fresnel.pdf
		vec3 fresnel = ( material.specularColor * t2.x + ( vec3( 1.0 ) - material.specularColor ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseColor * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometry.normal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometry.clearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecular += ccIrradiance * BRDF_GGX( directLight.direction, geometry.viewDir, geometry.clearcoatNormal, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecular += irradiance * BRDF_Sheen( directLight.direction, geometry.viewDir, geometry.normal, material.sheenColor, material.sheenRoughness );
	#endif
	#ifdef USE_IRIDESCENCE
		reflectedLight.directSpecular += irradiance * BRDF_GGX_Iridescence( directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness );
	#else
		reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, material.specularF90, material.roughness );
	#endif
    // Calculate subsurface scattering
    vec4 position = projectionMatrix * vec4(geometry.position, 1.0);
    vec2 uv = position.xy / position.w;
    uv = uv * 0.5 + 0.5;
    float thickness = texture2D(thicknessTexture, uv).r;
    vec3 scatteringHalf = normalize(directLight.direction + (geometry.normal * internalRoughness));
    float dotNLSubsurface = saturate( dot( geometry.viewDir, -scatteringHalf) );
    float specPow = mix(256.0, mix(1.0, 256.0, pow(1.0 - internalRoughness, 5.185)), pow(material.roughness, 0.1));
    vec3 subsurfaceIrradiance = scattering * pow(dotNLSubsurface, specPow) * BRDF_Lambert(directLight.color) * exp(-(1.0 / attenuationDistance) * thickness * (0.15 - 0.1 * internalRoughness) * scatteringAbsorption);
    reflectedLight.directSpecular += subsurfaceIrradiance;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in GeometricContext geometry, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecular += clearcoatRadiance * EnvironmentBRDF( geometry.clearcoatNormal, geometry.viewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecular += irradiance * material.sheenColor * IBLSheenBRDF( geometry.normal, geometry.viewDir, material.sheenRoughness );
	#endif
	// Both indirect specular and indirect diffuse light accumulate here
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometry.normal, geometry.viewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnel, material.roughness, singleScattering, multiScattering );
	#else
		computeMultiscattering( geometry.normal, geometry.viewDir, material.specularColor, material.specularF90, material.roughness, singleScattering, multiScattering );
	#endif
	vec3 totalScattering = singleScattering + multiScattering;
	vec3 diffuse = material.diffuseColor * ( 1.0 - max( max( totalScattering.r, totalScattering.g ), totalScattering.b ) );
	reflectedLight.indirectSpecular += radiance * singleScattering;
	reflectedLight.indirectSpecular += multiScattering * cosineWeightedIrradiance;
	reflectedLight.indirectDiffuse += diffuse * cosineWeightedIrradiance;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
// ref: https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}
`);
    }
    updateRenderTargets() {
        if (this.thicknessRenderTarget === null) {
            this.thicknessRenderTarget = new THREE.WebGLRenderTarget(this.renderTargetSize.x, this.renderTargetSize.y, {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                type: THREE.FloatType,
            });
            this.thicknessRenderTarget.depthTexture = new THREE.DepthTexture(this.thicknessRenderTarget, this.renderTargetSize.y, THREE.FloatType);
        }
        if (this.thicknessRenderTargetBlur === null) {
            this.thicknessRenderTargetBlur = new THREE.WebGLRenderTarget(this.renderTargetSize.x, this.renderTargetSize.y, {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                type: THREE.FloatType,
            });
        }
        if (this.thicknessRenderTargetDepth === null) {
            this.thicknessRenderTargetDepth = new THREE.WebGLRenderTarget(this.renderTargetSize.x, this.renderTargetSize.y, {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                type: THREE.FloatType,
            });
        }
        if (this.thicknessRenderTarget.width !== this.renderTargetSize.x || this.thicknessRenderTarget.height !== this.renderTargetSize.y) {
            this.thicknessRenderTarget.setSize(this.renderTargetSize.x, this.renderTargetSize.y);
            this.thicknessRenderTargetDepth.setSize(this.renderTargetSize.x, this.renderTargetSize.y);
            this.thicknessRenderTargetBlur.setSize(this.renderTargetSize.x, this.renderTargetSize.y);
        }
    }
}

export { MeshTranslucentMaterial };