/**
 * Quantum Demand Forecasting - 3D Plasma Core & Bloch Visualization
 * Built exclusively using native Three.js r160 features (no shaders, no external assets).
 */

import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

(function initQuantumScene() {
  const canvas = document.getElementById('scene');
  if (!canvas) return;

  // Check prefers-reduced-motion
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // WebGL Availability check and Fallback
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance',
    });
  } catch (err) {
    console.warn('WebGL initialization failed, falling back to CSS background:', err);
    canvas.style.display = 'none';
    return;
  }

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 7.5);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xfff4e6, 0.7);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffb703, 1.2);
  dirLight.position.set(5, 5, 5);
  scene.add(dirLight);

  // 1. Plasma Core: Wireframe Icosahedron + Inner Emissive Sphere
  const coreGroup = new THREE.Group();
  scene.add(coreGroup);

  const outerGeo = new THREE.IcosahedronGeometry(0.85, 1);
  const outerMat = new THREE.MeshBasicMaterial({
    color: 0xffb703,
    wireframe: true,
    transparent: true,
    opacity: 0.8,
  });
  const outerMesh = new THREE.Mesh(outerGeo, outerMat);
  coreGroup.add(outerMesh);

  const innerGeo = new THREE.SphereGeometry(0.48, 24, 24);
  const innerMat = new THREE.MeshStandardMaterial({
    color: 0xff5d3a,
    emissive: 0xff5d3a,
    emissiveIntensity: 0.9,
    roughness: 0.35,
  });
  const innerMesh = new THREE.Mesh(innerGeo, innerMat);
  coreGroup.add(innerMesh);

  // Helper for Soft Glow Sprite Texture
  function createGlowTexture(colorHex) {
    const c = document.createElement('canvas');
    c.width = 128;
    c.height = 128;
    const ctx = c.getContext('2d');
    const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    const col = new THREE.Color(colorHex);
    const rgb = `${Math.floor(col.r * 255)}, ${Math.floor(col.g * 255)}, ${Math.floor(col.b * 255)}`;
    grad.addColorStop(0, `rgba(${rgb}, 1)`);
    grad.addColorStop(0.3, `rgba(${rgb}, 0.5)`);
    grad.addColorStop(0.7, `rgba(${rgb}, 0.12)`);
    grad.addColorStop(1, `rgba(${rgb}, 0)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  }

  // 2. Four Qubit Spheres on an Orbiting Ring (radius 2.2)
  const RING_RADIUS = 2.2;
  const qubitColors = [0xffb703, 0xff5d3a, 0xe01e8c, 0xc8ff2e];
  const ringGroup = new THREE.Group();
  scene.add(ringGroup);

  const qubitMeshes = [];
  const qubitArrows = [];
  const UP_VECTOR = new THREE.Vector3(0, 1, 0);

  for (let i = 0; i < 4; i++) {
    const angle = (i * Math.PI * 2) / 4;
    const x = Math.cos(angle) * RING_RADIUS;
    const y = Math.sin(angle) * RING_RADIUS;

    const qHolder = new THREE.Group();
    qHolder.position.set(x, y, 0);

    // Qubit solid sphere
    const qGeo = new THREE.SphereGeometry(0.18, 16, 16);
    const qMat = new THREE.MeshStandardMaterial({
      color: qubitColors[i],
      emissive: qubitColors[i],
      emissiveIntensity: 0.7,
      roughness: 0.2,
    });
    const qMesh = new THREE.Mesh(qGeo, qMat);
    qHolder.add(qMesh);

    // Soft glow sprite
    const spriteMat = new THREE.SpriteMaterial({
      map: createGlowTexture(qubitColors[i]),
      transparent: true,
      blending: THREE.AdditiveBlending,
      opacity: 0.75,
    });
    const glowSprite = new THREE.Sprite(spriteMat);
    glowSprite.scale.set(0.9, 0.9, 1);
    qHolder.add(glowSprite);

    // 4. Bloch Arrow: ArrowHelper starting at qubit position pointing UP (|0>)
    const arrow = new THREE.ArrowHelper(UP_VECTOR, new THREE.Vector3(0, 0, 0), 0.75, qubitColors[i], 0.22, 0.12);
    qHolder.add(arrow);

    ringGroup.add(qHolder);
    qubitMeshes.push(qHolder);
    qubitArrows.push(arrow);
  }

  // 3. Entanglement: Line segments connecting qubit i to (i+1)%4
  const lineGeos = [];
  const lineMats = [];
  const lineGroup = new THREE.Group();
  ringGroup.add(lineGroup);

  for (let i = 0; i < 4; i++) {
    const next = (i + 1) % 4;
    const pts = [qubitMeshes[i].position.clone(), qubitMeshes[next].position.clone()];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({
      color: 0xff5d3a,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
    });
    const line = new THREE.Line(geo, mat);
    lineGroup.add(line);
    lineGeos.push(geo);
    lineMats.push(mat);
  }

  // 5. Particles: 1200 points (500 on screens < 768px)
  const isMobile = window.innerWidth < 768;
  const particleCount = isMobile ? 500 : 1200;
  const particleGeo = new THREE.BufferGeometry();
  const particlePositions = new Float32Array(particleCount * 3);
  const particleColors = new Float32Array(particleCount * 3);
  const palette = [new THREE.Color(0xffb703), new THREE.Color(0xff5d3a), new THREE.Color(0xe01e8c), new THREE.Color(0xfff4e6)];

  for (let i = 0; i < particleCount; i++) {
    particlePositions[i * 3] = (Math.random() - 0.5) * 14;
    particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 14;
    particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 8 - 2;

    const col = palette[Math.floor(Math.random() * palette.length)];
    particleColors[i * 3] = col.r;
    particleColors[i * 3 + 1] = col.g;
    particleColors[i * 3 + 2] = col.b;
  }

  particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
  particleGeo.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));

  const particleMat = new THREE.PointsMaterial({
    size: 0.05,
    vertexColors: true,
    transparent: true,
    opacity: 0.65,
    blending: THREE.AdditiveBlending,
  });
  const particles = new THREE.Points(particleGeo, particleMat);
  scene.add(particles);

  // 6. Camera Mouse Parallax
  let mouseX = 0;
  let mouseY = 0;
  let targetMouseX = 0;
  let targetMouseY = 0;

  window.addEventListener('mousemove', (e) => {
    targetMouseX = (e.clientX / window.innerWidth - 0.5) * 0.9;
    targetMouseY = (e.clientY / window.innerHeight - 0.5) * 0.9;
  });

  // Window Resize Handler
  function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  }
  window.addEventListener('resize', onWindowResize);

  // Tab Visibility Handler (pause rendering when tab is hidden)
  let isTabVisible = true;
  document.addEventListener('visibilitychange', () => {
    isTabVisible = !document.hidden;
  });

  // 7. Sequence State Machine for playQuantumSequence(data)
  let isSequenceActive = false;
  let sequenceStartTime = 0;
  let sequenceData = null;

  // Safe vector normalization helper
  function toNormalizedVector(arr) {
    if (!Array.isArray(arr) || arr.length < 3) return UP_VECTOR.clone();
    const v = new THREE.Vector3(arr[0], arr[1], arr[2]);
    if (v.lengthSq() < 1e-5) return UP_VECTOR.clone();
    return v.normalize();
  }

  window.playQuantumSequence = function (data) {
    if (!data || !Array.isArray(data.bloch_encoded) || !Array.isArray(data.bloch_final)) return;
    sequenceData = data;
    sequenceStartTime = performance.now();
    isSequenceActive = true;

    // If reduced motion is requested, instantly set final directions without animation
    if (prefersReducedMotion) {
      for (let i = 0; i < 4; i++) {
        const finalVec = toNormalizedVector(data.bloch_final[i]);
        qubitArrows[i].setDirection(finalVec);
      }
      innerMat.emissiveIntensity = 1.6;
      setTimeout(() => {
        innerMat.emissiveIntensity = 0.9;
      }, 500);
      renderer.render(scene, camera);
    }
  };

  // Main Render Animation Loop
  let clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    if (!isTabVisible) return;

    const delta = clock.getDelta();
    const elapsedTime = clock.getElapsedTime();

    if (prefersReducedMotion) {
      // Just render static scene
      renderer.render(scene, camera);
      return;
    }

    // Camera parallax lerp
    mouseX = THREE.MathUtils.lerp(mouseX, targetMouseX, 0.05);
    mouseY = THREE.MathUtils.lerp(mouseY, targetMouseY, 0.05);
    camera.position.x = mouseX;
    camera.position.y = -mouseY;
    camera.lookAt(0, 0, 0);

    // Plasma core slow rotation & gentle scale pulse
    coreGroup.rotation.x += delta * 0.35;
    coreGroup.rotation.y += delta * 0.45;
    const pulseScale = 1.0 + 0.07 * Math.sin(elapsedTime * 2.5);
    innerMesh.scale.set(pulseScale, pulseScale, pulseScale);

    // Default ring orbit & entanglement line pulsing
    let ringSpeed = 0.25;
    const linePulse = 0.35 + 0.25 * Math.sin(elapsedTime * 3.0);
    lineMats.forEach((mat) => {
      mat.opacity = linePulse;
    });

    // Handle Quantum Prediction Sequence State Machine
    if (isSequenceActive && sequenceData) {
      const seqElapsed = (performance.now() - sequenceStartTime) / 1000;

      if (seqElapsed <= 1.0) {
        // Phase 1 (0 - 1.0s): Arrows rotate from UP (0, 1, 0) to bloch_encoded[i]
        const progress = Math.min(seqElapsed / 1.0, 1.0);
        const ease = 1 - Math.pow(1 - progress, 3);
        for (let i = 0; i < 4; i++) {
          const encVec = toNormalizedVector(sequenceData.bloch_encoded[i]);
          const currentVec = new THREE.Vector3().copy(UP_VECTOR).lerp(encVec, ease).normalize();
          qubitArrows[i].setDirection(currentVec);
        }
      } else if (seqElapsed <= 2.2) {
        // Phase 2 (1.0 - 2.2s): Entanglement lines brighten and ring spins faster
        const subElapsed = (seqElapsed - 1.0) / 1.2;
        ringSpeed = 0.75 + 0.5 * Math.sin(subElapsed * Math.PI);
        lineMats.forEach((mat) => {
          mat.opacity = 0.95;
        });
      } else if (seqElapsed <= 3.4) {
        // Phase 3 (2.2 - 3.4s): Arrows lerp from bloch_encoded to bloch_final
        const subProgress = Math.min((seqElapsed - 2.2) / 1.2, 1.0);
        const ease = 1 - Math.pow(1 - subProgress, 3);
        for (let i = 0; i < 4; i++) {
          const encVec = toNormalizedVector(sequenceData.bloch_encoded[i]);
          const finVec = toNormalizedVector(sequenceData.bloch_final[i]);
          const currentVec = new THREE.Vector3().copy(encVec).lerp(finVec, ease).normalize();
          qubitArrows[i].setDirection(currentVec);
        }
      } else if (seqElapsed <= 4.2) {
        // Phase 4 (3.4s+): Core flashes brighter once and settles
        const flashProgress = (seqElapsed - 3.4) / 0.8;
        if (flashProgress < 0.4) {
          innerMat.emissiveIntensity = 0.9 + 1.4 * (flashProgress / 0.4);
        } else {
          innerMat.emissiveIntensity = 2.3 - 1.4 * ((flashProgress - 0.4) / 0.6);
        }
      } else {
        // Sequence finishes and settles
        innerMat.emissiveIntensity = 0.9;
        isSequenceActive = false;
      }
    }

    // Apply ring orbit rotation
    ringGroup.rotation.z += delta * ringSpeed;

    // Slowly drift particles
    particles.rotation.y += delta * 0.05;
    particles.rotation.x += delta * 0.02;

    renderer.render(scene, camera);
  }

  // Start loop
  animate();
})();
