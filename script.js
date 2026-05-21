import * as tf from "https://esm.sh/@tensorflow/tfjs";
import * as gymjs from "https://esm.sh/@unexploredtest/gymjs@0.1.17";
import * as rljs from "https://esm.sh/@unexploredtest/rljs@0.0.4";

import {setWasmPaths} from "https://esm.sh/@tensorflow/tfjs-backend-wasm";
setWasmPaths("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/")

/*
  ==========================================
  CONFIG
  ==========================================
*/

const TOTAL_CYCLES = 20; // <- Change this
const TRAINING_STEPS = 5_000;
const TRAINING_TIME_MS = 2500;
const VISUALIZATION_TIME_MS = 3500;

/*
  ==========================================
  ELEMENTS
  ==========================================
*/

const statusText = document.getElementById("statusText");
const cycleText = document.getElementById("cycleText");
const progressFill = document.getElementById("progressFill");

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

/*
  ==========================================
  RL
  ==========================================
*/

let cartpoleEnv = new gymjs.envs.classic_control.CartPoleEnv();
cartpoleEnv = new gymjs.wrappers.TimeLimit(cartpoleEnv, 2000);

let ppo = new rljs.PPO(cartpoleEnv, {}, {}, true);


/*
  ==========================================
  CANVAS SETUP
  ==========================================
*/

function resizeCanvas() {
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
}

resizeCanvas();
window.addEventListener("resize", resizeCanvas);

/*
  ==========================================
  DEMO VISUALIZATION
  Replace this with your actual agent render
  ==========================================
*/

async function evaluateAgent() {

  let envRendered =
    new gymjs.envs.classic_control.CartPoleEnv(false, "human", canvas);

  envRendered = new gymjs.wrappers.TimeLimit(envRendered, 2000);

  let [obs] = envRendered.reset();
  let finished = false;

  while (!finished) {

    let action = ppo.predict(obs, true);

    let [obsNew, reward, done, trun] =
      await envRendered.step(action.dataSync()[0]);

    obs = obsNew;
    finished = done || trun;
  }

  envRendered.close();
}


let animationFrame;

/*
  ==========================================
  TRAINING LOOP
  ==========================================
*/

async function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function startTraining() {
  await tf.setBackend("wasm");
  await tf.ready();

  for (let cycle = 1; cycle <= TOTAL_CYCLES; cycle++) {

    /*
      TRAINING PHASE
    */

    statusText.innerText =
      `Training agent for ${TRAINING_STEPS} steps...`;

    cycleText.innerText =
      `Training...`;

    progressFill.style.width =
      `${(cycle / TOTAL_CYCLES) * 100}%`;

    cancelAnimationFrame(animationFrame);

    // Optional loading visual
    drawLoadingState(cycle);

    // Here you can run your actual training logic
    await delay(100);
    await ppo.learn(1_000, null, false);

    /*
      VISUALIZATION PHASE
    */

    statusText.innerText =
      `Evaluating performance after ${TRAINING_STEPS * cycle} steps...`;

    cancelAnimationFrame(animationFrame);
    ctx.clearRect(0, 0, canvas.width, canvas.height);


    let originalWidth = canvas.width;
    let originalHeight = canvas.height;
    await evaluateAgent();
    canvas.width = originalWidth;
    canvas.height = originalHeight;
  }

  /*
    DONE
  */

  cancelAnimationFrame(animationFrame);

  statusText.innerText =
    "Training complete.";

  drawFinishedScreen();
}

/*
  ==========================================
  LOADING SCREEN
  ==========================================
*/

function drawLoadingState(cycle) {

  function animate() {

    const t = Date.now() * 0.002;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    // rotating rings
    for (let i = 0; i < 3; i++) {

      ctx.beginPath();

      ctx.arc(
        centerX,
        centerY,
        70 + i * 35,
        t + i,
        t + Math.PI * 1.5 + i
      );

      ctx.strokeStyle =
        i % 2 === 0
          ? "rgba(124,92,255,0.8)"
          : "rgba(0,212,255,0.8)";

      ctx.lineWidth = 5;
      ctx.stroke();
    }

    ctx.font = "600 22px Inter";
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.textAlign = "center";

    ctx.fillText(
      `Training...`,
      centerX,
      centerY + 10
    );

    animationFrame =
      requestAnimationFrame(animate);
  }

  animate();
}

/*
  ==========================================
  FINISHED SCREEN
  ==========================================
*/

function drawFinishedScreen() {

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const gradient = ctx.createLinearGradient(
    0,
    0,
    canvas.width,
    canvas.height
  );

  gradient.addColorStop(0, "#7c5cff");
  gradient.addColorStop(1, "#00d4ff");

  ctx.fillStyle = gradient;

  ctx.font = "700 54px Inter";
  ctx.textAlign = "center";

  ctx.fillText(
    "Training Complete",
    canvas.width / 2,
    canvas.height / 2
  );
}

/*
  ==========================================
  START
  ==========================================
*/

startTraining();
