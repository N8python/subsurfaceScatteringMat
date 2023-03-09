import * as THREE from 'https://cdn.skypack.dev/three@0.150.0';
import { EffectComposer } from 'https://unpkg.com/three@0.150.0/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'https://unpkg.com/three@0.150.0/examples/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from 'https://unpkg.com/three@0.150.0/examples/jsm/postprocessing/ShaderPass.js';
import { SMAAPass } from 'https://unpkg.com/three@0.150.0/examples/jsm/postprocessing/SMAAPass.js';
import { GammaCorrectionShader } from 'https://unpkg.com/three@0.150.0/examples/jsm/shaders/GammaCorrectionShader.js';
import { EffectShader } from "./EffectShader.js";
import { OrbitControls } from 'https://unpkg.com/three@0.150.0/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from "https://unpkg.com/three@0.150.0/examples/jsm/loaders/OBJLoader.js";
import { GLTFLoader } from "https://unpkg.com/three@0.150.0/examples/jsm/loaders/GLTFLoader.js";
import * as BufferGeometryUtils from "https://unpkg.com/three@0.150.0/examples/jsm/utils/BufferGeometryUtils.js";
import { FullScreenQuad } from "https://unpkg.com/three@0.150.0/examples/jsm/postprocessing/Pass.js";
import { AssetManager } from './AssetManager.js';
import { Stats } from "./stats.js";
async function main() {
    // Setup basic renderer, controls, and profiler
    const clientWidth = window.innerWidth * 0.99;
    const clientHeight = window.innerHeight * 0.98;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, clientWidth / clientHeight, 0.1, 1000);
    camera.position.set(50, 75, 50);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(clientWidth, clientHeight);
    document.body.appendChild(renderer.domElement);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 25, 0);
    const stats = new Stats();
    stats.showPanel(0);
    document.body.appendChild(stats.dom);
    // Setup scene
    // Skybox
    const environment = new THREE.CubeTextureLoader().load([
        "skybox/Box_Right.bmp",
        "skybox/Box_Left.bmp",
        "skybox/Box_Top.bmp",
        "skybox/Box_Bottom.bmp",
        "skybox/Box_Front.bmp",
        "skybox/Box_Back.bmp"
    ]);
    environment.encoding = THREE.sRGBEncoding;
    scene.background = environment;
    // Lighting
    const ambientLight = new THREE.AmbientLight(new THREE.Color(1.0, 1.0, 1.0), 0.25);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.35);
    directionalLight.position.set(150, 200, 50);
    // Shadows
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;
    directionalLight.shadow.camera.left = -75;
    directionalLight.shadow.camera.right = 75;
    directionalLight.shadow.camera.top = 75;
    directionalLight.shadow.camera.bottom = -75;
    directionalLight.shadow.camera.near = 0.1;
    directionalLight.shadow.camera.far = 500;
    directionalLight.shadow.bias = -0.001;
    directionalLight.shadow.blurSamples = 8;
    directionalLight.shadow.radius = 4;
    scene.add(directionalLight);
    // Add 6 point lights
    const colors = [
        new THREE.Color(1.0, 0.0, 0.0),
        new THREE.Color(0.0, 1.0, 0.0),
        new THREE.Color(0.0, 0.0, 1.0),
        new THREE.Color(1.0, 1.0, 0.0),
        new THREE.Color(1.0, 0.0, 1.0),
        new THREE.Color(0.0, 1.0, 1.0)
    ];
    const lights = [];
    for (let i = 0; i < 6; i++) {
        let color = colors[i % 6];
        const pointLight = new THREE.PointLight(color, 0.25);
        pointLight.position.set(40 * Math.random() - 20, 5 + 10 * Math.random(), 40 * Math.random() - 20);
        scene.add(pointLight);
        pointLight.add(new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshBasicMaterial({ color: color })));
        lights.push(pointLight);
    }
    // Objects
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(100, 100).applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI / 2)), new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, roughness: 0, envMap: environment, color: new THREE.Color(0.5, 0.5, 0.5) }));
    ground.castShadow = true;
    ground.receiveShadow = true;
    scene.add(ground);
    /*const box = new THREE.Mesh(new THREE.BoxGeometry(10, 10, 10), new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, color: new THREE.Color(1.0, 0.0, 0.0) }));
    box.castShadow = true;
    box.receiveShadow = true;
    box.position.y = 5.01;
    scene.add(box);
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(6.25, 32, 32), new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, envMap: environment, metalness: 1.0, roughness: 0.25 }));
    sphere.position.y = 7.5;
    sphere.position.x = 25;
    sphere.position.z = 25;
    sphere.castShadow = true;
    sphere.receiveShadow = true;
    scene.add(sphere);*/
    // Build postprocessing stack
    // Render Targets
    const defaultTexture = new THREE.WebGLRenderTarget(clientWidth, clientHeight, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.NearestFilter
    });
    defaultTexture.depthTexture = new THREE.DepthTexture(clientWidth, clientHeight, THREE.FloatType);
    const thicknessTexture = new THREE.WebGLRenderTarget(clientWidth, clientHeight, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        type: THREE.FloatType,
    });
    thicknessTexture.depthTexture = new THREE.DepthTexture(clientWidth, clientHeight, THREE.FloatType);
    const thicknessBlurTexture = thicknessTexture.clone();
    const thicknessDepth = new THREE.WebGLRenderTarget(clientWidth, clientHeight, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        type: THREE.FloatType,
    });
    const thicknessHBlur = new FullScreenQuad(new THREE.ShaderMaterial({
        uniforms: {
            tDiffuse: { value: thicknessTexture.texture },
            tDepth: { value: thicknessDepth.texture },
            uResolution: { value: new THREE.Vector2(clientWidth, clientHeight) },
            size: { value: 4.0 },
            stride: { value: 8.0 }
        },
        vertexShader: /*glsl*/ `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
        `,
        fragmentShader: /*glsl*/ `
        uniform sampler2D tDiffuse;
        uniform sampler2D tDepth;
        uniform float size;
        uniform float stride;
        uniform vec2 uResolution;
        varying vec2 vUv;
        highp float linearize_depth(highp float d, highp float zNear,highp float zFar)
        {
            highp float z_n = 2.0 * d - 1.0;
            return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
        }
        void main() {
            vec2 uv = vUv;
            float depth = linearize_depth(texture2D(tDepth, uv).x, 0.1, 1000.0);
            // Attenuate the stride based on the depth, so that the blur is larger near the camera
            float updatedStride = stride * (1.0 / (1.0 + (depth - 0.1)));
            vec2 invResolution = 1.0 / uResolution;
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
                color += texel.rgb * weight;
                total += weight;
            }
            color /= total;
            gl_FragColor = vec4(color, 1.0);
        }`
    }));
    const thicknessVBlur = new FullScreenQuad(new THREE.ShaderMaterial({
        uniforms: {
            tDiffuse: { value: thicknessBlurTexture.texture },
            tDepth: { value: thicknessDepth.texture },
            uResolution: { value: new THREE.Vector2(clientWidth, clientHeight) },
            size: { value: 4.0 },
            stride: { value: 8.0 }
        },
        vertexShader: /*glsl*/ `
        varying vec2 vUv;
        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
        `,
        fragmentShader: /*glsl*/ `
        uniform sampler2D tDiffuse;
        uniform sampler2D tDepth;
        uniform float size;
        uniform float stride;
        uniform vec2 uResolution;
        varying vec2 vUv;
        highp float linearize_depth(highp float d, highp float zNear,highp float zFar)
        {
            highp float z_n = 2.0 * d - 1.0;
            return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
        }
        void main() {
            vec2 uv = vUv;
            float depth = linearize_depth(texture2D(tDepth, uv).x, 0.1, 1000.0);
            float updatedStride = stride * (1.0 / (1.0 + (depth - 0.1)));
            vec2 invResolution = 1.0 / uResolution;
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
                color += texel.rgb * weight;
                total += weight;
            }
            color /= total;
            gl_FragColor = vec4(color, 1.0);

        }`

    }));
    const depthBlitQuad = new FullScreenQuad(new THREE.ShaderMaterial({
        uniforms: {
            tDiffuse: { value: thicknessTexture.depthTexture },
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
        uniform sampler2D tDiffuse;
        highp float linearize_depth(highp float d, highp float zNear,highp float zFar)
        {
            highp float z_n = 2.0 * d - 1.0;
            return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
        }
        void main() {
            gl_FragColor = vec4(texture2D(tDiffuse, vUv).x, 0.0, 0.0, 1.0);
        }
        `
    }));
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
    const depthMat = new THREE.MeshBasicMaterial({
        colorWrite: false,
        depthWrite: true,
        side: THREE.DoubleSide,
    });
    let buddha = await new OBJLoader().loadAsync("https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/happy.obj");
    // buddha = buddha.children[0];
    buddha.traverse((child) => {
        if (child.isMesh) {
            buddha = child;
        }
    });
    buddha.geometry.deleteAttribute('normal');
    buddha.geometry = BufferGeometryUtils.mergeVertices(buddha.geometry);
    buddha.geometry.computeVertexNormals();
    buddha.updateMatrixWorld();
    buddha.geometry.applyMatrix4(buddha.matrixWorld);
    buddha.geometry.center();
    buddha.geometry.scale(100, 100, 100);
    const translucentMesh = new THREE.Mesh(buddha.geometry, new THREE.MeshPhysicalMaterial({ side: THREE.DoubleSide, envMap: environment, transmission: 1.0, roughness: 0.5, ior: 1.5, attenuationColor: new THREE.Color(0.9, 0.6, 0.3), attenuationDistance: 0.33, dithering: true, thickness: 2.0 }));

    translucentMesh.position.y = new THREE.Box3().setFromObject(translucentMesh, true).getSize(new THREE.Vector3()).y / 2;
    translucentMesh.material.onBeforeCompile = (shader) => {
        shader.uniforms.thicknessTexture = { value: thicknessTexture.texture };

        shader.fragmentShader = "uniform sampler2D thicknessTexture;\nuniform mat4 projectionMatrix;\nuniform float attenuationDistance;\n        " + shader.fragmentShader.replace(
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
    vec3 scatteringHalf = normalize(directLight.direction + (geometry.normal * 0.5));
    float dotNLSubsurface = saturate( dot( geometry.viewDir, -scatteringHalf) );
    float specPow = 256.0 - 248.0 * pow(material.roughness, 0.1);
    vec3 subsurfaceIrradiance = pow(dotNLSubsurface, specPow) * BRDF_Lambert(directLight.color) * exp(-(1.0 / attenuationDistance) * thickness * 0.1);
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
`).replace(
            "#include <lights_fragment_begin>",
            /*glsl*/
            `
            GeometricContext geometry;
            geometry.position = - vViewPosition;
            geometry.normal = normal;
            geometry.viewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
            #ifdef USE_CLEARCOAT
                geometry.clearcoatNormal = clearcoatNormal;
            #endif
            #ifdef USE_IRIDESCENCE
                float dotNVi = saturate( dot( normal, geometry.viewDir ) );
                if ( material.iridescenceThickness == 0.0 ) {
                    material.iridescence = 0.0;
                } else {
                    material.iridescence = saturate( material.iridescence );
                }
                if ( material.iridescence > 0.0 ) {
                    material.iridescenceFresnel = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
                    // Iridescence F0 approximation
                    material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
                }
            #endif
            IncidentLight directLight;
            #if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
                PointLight pointLight;
                #if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
                PointLightShadow pointLightShadow;
                #endif
                #pragma unroll_loop_start
                for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
                    pointLight = pointLights[ i ];
                    getPointLightInfo( pointLight, geometry, directLight );
                    #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS )
                    pointLightShadow = pointLightShadows[ i ];
                    directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
                    #endif
                    RE_Direct( directLight, geometry, material, reflectedLight );
                }
                #pragma unroll_loop_end
            #endif
            #if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
                SpotLight spotLight;
                vec4 spotColor;
                vec3 spotLightCoord;
                bool inSpotLightMap;
                #if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
                SpotLightShadow spotLightShadow;
                #endif
                #pragma unroll_loop_start
                for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
                    spotLight = spotLights[ i ];
                    getSpotLightInfo( spotLight, geometry, directLight );
                    // spot lights are ordered [shadows with maps, shadows without maps, maps without shadows, none]
                    #if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
                    #define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
                    #elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
                    #define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
                    #else
                    #define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
                    #endif
                    #if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
                        spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
                        inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
                        spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
                        directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
                    #endif
                    #undef SPOT_LIGHT_MAP_INDEX
                    #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
                    spotLightShadow = spotLightShadows[ i ];
                    directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
                    #endif
                    RE_Direct( directLight, geometry, material, reflectedLight );
                }
                #pragma unroll_loop_end
            #endif
            #if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
                DirectionalLight directionalLight;
                #if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
                DirectionalLightShadow directionalLightShadow;
                #endif
                #pragma unroll_loop_start
                for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
                    directionalLight = directionalLights[ i ];
                    getDirectionalLightInfo( directionalLight, geometry, directLight );
                    #if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
                    directionalLightShadow = directionalLightShadows[ i ];
                    directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
                    #endif
                    RE_Direct( directLight, geometry, material, reflectedLight );
                }
                #pragma unroll_loop_end
            #endif
            #if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
                RectAreaLight rectAreaLight;
                #pragma unroll_loop_start
                for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
                    rectAreaLight = rectAreaLights[ i ];
                    RE_Direct_RectArea( rectAreaLight, geometry, material, reflectedLight );
                }
                #pragma unroll_loop_end
            #endif
            #if defined( RE_IndirectDiffuse )
                vec3 iblIrradiance = vec3( 0.0 );
                vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
                irradiance += getLightProbeIrradiance( lightProbe, geometry.normal );
                #if ( NUM_HEMI_LIGHTS > 0 )
                    #pragma unroll_loop_start
                    for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
                        irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometry.normal );
                    }
                    #pragma unroll_loop_end
                #endif
            #endif
            #if defined( RE_IndirectSpecular )
                vec3 radiance = vec3( 0.0 );
                vec3 clearcoatRadiance = vec3( 0.0 );
            #endif
            `
        )
    }
    translucentMesh.castShadow = true;
    //translucentMesh.receiveShadow = true;
    scene.add(translucentMesh);

    // Post Effects
    const composer = new EffectComposer(renderer);
    const smaaPass = new SMAAPass(clientWidth, clientHeight);
    const effectPass = new ShaderPass(EffectShader);
    composer.addPass(effectPass);
    composer.addPass(new ShaderPass(GammaCorrectionShader));
    composer.addPass(smaaPass);
    const roughnessBlurSize = 16.0;
    const clock = new THREE.Clock();

    function animate() {
        const delta = clock.getDelta();
        translucentMesh.material.roughness = Math.cos(performance.now() / 1000) * 0.5 + 0.5;
        // Rotate point lights:
        for (let i = 0; i < lights.length; i++) {
            const light = lights[i];
            const mag = Math.max(Math.hypot(light.position.x, light.position.z), 5);
            const angle = Math.atan2(light.position.z, light.position.x) + delta;
            light.position.x = mag * Math.cos(angle);
            light.position.z = mag * Math.sin(angle);
        }

        //translucentMesh.rotation.y += 0.01;
        renderer.setRenderTarget(thicknessTexture);
        renderer.setClearAlpha(0.0);
        renderer.clear();
        renderer.autoClear = false;
        const oldMat = translucentMesh.material;
        translucentMesh.material = thicknessMaterial;
        renderer.render(translucentMesh, camera);
        translucentMesh.material = depthMat;
        renderer.render(translucentMesh, camera);
        translucentMesh.material = oldMat;
        renderer.setClearAlpha(1.0);
        renderer.autoClear = true;
        renderer.setRenderTarget(thicknessDepth);
        depthBlitQuad.render(renderer);
        thicknessHBlur.material.uniforms.stride.value = roughnessBlurSize * 8.0 * translucentMesh.material.roughness;
        thicknessVBlur.material.uniforms.stride.value = roughnessBlurSize * 8.0 * translucentMesh.material.roughness;
        renderer.setRenderTarget(thicknessBlurTexture);
        thicknessHBlur.render(renderer);
        renderer.setRenderTarget(thicknessTexture);
        thicknessVBlur.render(renderer);
        thicknessHBlur.material.uniforms.stride.value = roughnessBlurSize * 1.0 * translucentMesh.material.roughness;
        thicknessVBlur.material.uniforms.stride.value = roughnessBlurSize * 1.0 * translucentMesh.material.roughness;
        renderer.setRenderTarget(thicknessBlurTexture);
        thicknessHBlur.render(renderer);
        renderer.setRenderTarget(thicknessTexture);
        thicknessVBlur.render(renderer);
        renderer.setRenderTarget(defaultTexture);
        renderer.clear();
        renderer.render(scene, camera);
        effectPass.uniforms["sceneDiffuse"].value = defaultTexture.texture;
        composer.render();
        controls.update();
        stats.update();
        requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
}
main();