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
import { GUI } from 'https://unpkg.com/three@0.150.0/examples/jsm/libs/lil-gui.module.min.js';
import { AssetManager } from './AssetManager.js';
import { MeshTranslucentMaterial } from './TranslucentMaterial.js';
import { Stats } from "./stats.js";
async function main() {
    // Setup basic renderer, controls, and profiler
    const clientWidth = window.innerWidth;
    const clientHeight = window.innerHeight;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, clientWidth / clientHeight, 0.1, 1000);
    camera.position.set(50, 75, 50);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(clientWidth, clientHeight);
    window.addEventListener('resize', () => {
        const clientWidth = window.innerWidth;
        const clientHeight = window.innerHeight;
        renderer.setSize(clientWidth, clientHeight);
        camera.aspect = clientWidth / clientHeight;
        camera.updateProjectionMatrix();
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputEncoding = THREE.sRGBEncoding;
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
    const rSize = renderer.getDrawingBufferSize(new THREE.Vector2());
    rSize.x = Math.floor(rSize.x / 2);
    rSize.y = Math.floor(rSize.y / 2);
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
    let buddha = await new OBJLoader().loadAsync("https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/happy.obj");
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
    const translucentMesh = new THREE.Mesh(buddha.geometry, new MeshTranslucentMaterial({ side: THREE.DoubleSide, envMap: environment, transmission: 1.0, roughness: 0.5, ior: 1.5, attenuationColor: new THREE.Color(0.9, 0.6, 0.3), attenuationDistance: 0.33, dithering: true, thickness: 2.0 }));
    translucentMesh.position.y = new THREE.Box3().setFromObject(translucentMesh, true).getSize(new THREE.Vector3()).y / 2;
    translucentMesh.castShadow = true;
    //translucentMesh.receiveShadow = true;
    scene.add(translucentMesh);

    const clock = new THREE.Clock();
    const gui = new GUI();
    const effectController = {
        roughness: 0.5,
        internalRoughness: 0.5,
        scatteringAbsorption: 1.0,
        scattering: 1.0,
        roughnessBlurScale: 16.0,
        resolutionScale: 0.5,
        attenuationDistance: 0.33,
        attenuationColor: [0.9, 0.6, 0.3]
    }
    gui.add(effectController, "roughness", 0.0, 1.0, 0.01).onChange((value) => {
        translucentMesh.material.roughness = value;
    });
    gui.add(effectController, "internalRoughness", 0.0, 1.0, 0.01).onChange((value) => {
        translucentMesh.material.internalRoughness = value;
    });
    gui.add(effectController, "scatteringAbsorption", 0.0, 1.0, 0.01).onChange((value) => {
        translucentMesh.material.scatteringAbsorption = value;
    });
    gui.add(effectController, "scattering", 0.0, 1.0, 0.01).onChange((value) => {
        translucentMesh.material.scattering = value;
    });
    gui.add(effectController, "roughnessBlurScale", 0.0, 32.0, 0.01).onChange((value) => {
        translucentMesh.material.roughnessBlurScale = value;
    });
    gui.add(effectController, "resolutionScale", 0.25, 1.0, 0.01).onChange((value) => {
        translucentMesh.material.resolutionScale = value;
    });
    gui.add(effectController, "attenuationDistance", 0.0, 1.0, 0.01).onChange((value) => {
        translucentMesh.material.attenuationDistance = value;
    });
    gui.addColor(effectController, "attenuationColor").onChange((value) => {
        translucentMesh.material.attenuationColor = new THREE.Color(value[0], value[1], value[2]);
    });



    function animate() {
        const delta = clock.getDelta();
        // Rotate point lights:
        /*for (let i = 0; i < lights.length; i++) {
            const light = lights[i];
            const mag = Math.max(Math.hypot(light.position.x, light.position.z), 5);
            const angle = Math.atan2(light.position.z, light.position.x) + delta;
            light.position.x = mag * Math.cos(angle);
            light.position.z = mag * Math.sin(angle);
        }*/
        renderer.render(scene, camera);
        controls.update();
        stats.update();
        requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
}
main();