const cW = 400, cH = 600;
const strokeW = 10;
const barLen = 40;
const barSpeed = 6;

let ballYSpeed = 3;
let ballXSpeed = 5;

const MAX_ABS_SPEED = 5;

let DETECT_TRAECTORY_TRAIN_TEMP_STRUCT = {};
let DETECT_TRAECTORY_TRAIN_SEQ = [];
let DETECT_GOAL_MODEL;
const collectDetectingData = false;

const lWall = {
  left: 0,
  top: 0,
  width: strokeW,
  height: cH
}

const rWall = {
  left: cW - strokeW,
  top: 0,
  width: strokeW,
  height: cH
}

const player2 = {
  left: cW / 2 - barLen / 2,
  top: cH - strokeW,
  width: barLen,
  height: strokeW
}

const player1 = {
  left: cW / 2 - barLen / 2,
  top: 0,
  width: barLen,
  height: strokeW
}

const ball = {
  left: cW / 2 - strokeW / 2,
  top: cH / 2 - strokeW / 2,
  width: strokeW,
  height: strokeW
}

const trainBar = {
  left: 0,
  top: cH - strokeW,
  width: cW,
  height: strokeW
}

function setup() {
  createCanvas(cW, cH);

  (async () => {
    DETECT_GOAL_MODEL = await tf.loadLayersModel('http://localhost:8080/detect-goal-pp-v0/model.json')
  })()
}

function draw() {
  background('#0072a9');

  correctBallSpeedIfCollision();
  updateBallPosition();
  updatePlayer1Position(DETECT_GOAL_MODEL);

  checkGameEnd();

  if (
    keyIsDown(RIGHT_ARROW)
  ) {
    updatePlayer2BarPosition(1)
  }

  if (
    keyIsDown(LEFT_ARROW)
  ) {
    updatePlayer2BarPosition(-1)
  }

  if (
    keyIsDown(65)
  ) {
    updatePlayer1BarPosition(-1)
  }

  if (
    keyIsDown(68)
  ) {
    updatePlayer1BarPosition(1)
  }

  drawWalls();

  if (!collectDetectingData)
    drawPlayer1Bar(DETECT_GOAL_MODEL);

  drawPlayer2Bar();

  drawBall();

  if (collectDetectingData) {
    drawTrainBar()
  }
}

function drawPredictedGoal() {
  const inputs = [ballXSpeed / MAX_ABS_SPEED, ballYSpeed / MAX_ABS_SPEED, ball.left / cW, ball.top / cH];

  const xs = tf.tensor2d([inputs])

  const ys = DETECT_GOAL_MODEL.predict(xs);

  const resultNormalizedLeft = ys.dataSync();
  const left = resultNormalizedLeft * cW;

  const predictedGoal = {
    left: left,
    top: 0,
    width: strokeW,
    height: 3 * strokeW,
  }

  const green = '#0f0';

  fill(green);
  stroke(green);
  rect(predictedGoal.left, predictedGoal.top, predictedGoal.width, predictedGoal.height);
}

function keyPressed() {
  loop();

  if (keyCode == RIGHT_ARROW) {
    updatePlayer2BarPosition(1)
  }

  if (keyCode == LEFT_ARROW) {
    updatePlayer2BarPosition(-1)
  }

  if (key == 'a') {
    updatePlayer1BarPosition(-1)
  }

  if (key == 'd') {
    updatePlayer1BarPosition(1)
  }
}

function getNewLeftForPlayer(dir, oldLeft) {
  const newLeft = oldLeft + (dir * barSpeed);

  if (newLeft > strokeW && (newLeft + barLen) < cW - strokeW) {
    return newLeft
  } else {
    return dir < 0 ? strokeW + 1 : cW - strokeW - barLen - 1
  }
}

function updatePlayer1BarPosition(dir) {
  player1.left = getNewLeftForPlayer(dir, player1.left)
}

function updatePlayer2BarPosition(dir) {
  player2.left = getNewLeftForPlayer(dir, player2.left)
}

function checkGameEnd() {
  if (ball.top <= 0 || (ball.top + ball.height >= cH)) {
    if (!collectDetectingData) {
      noLoop()
    } else {
      if (DETECT_TRAECTORY_TRAIN_TEMP_STRUCT.ballXSpeed) { // not first time
        DETECT_TRAECTORY_TRAIN_TEMP_STRUCT.resultBallLeft = ball.left;
        DETECT_TRAECTORY_TRAIN_TEMP_STRUCT.resultBalltop = ball.top;

        DETECT_TRAECTORY_TRAIN_SEQ.push(DETECT_TRAECTORY_TRAIN_TEMP_STRUCT);

        DETECT_TRAECTORY_TRAIN_TEMP_STRUCT = {};
      }

      if (DETECT_TRAECTORY_TRAIN_SEQ.length == 500) {
        noLoop();

        saveJSON({
          width: cW,
          height: cH,
          data: DETECT_TRAECTORY_TRAIN_SEQ
        }, `${cW}-${cH}-500.json`);
      }
    }

    // ballXSpeed = random(-1, 1) < 0 ? random(-3, -7) : random(3, 7);
    // ballYSpeed = random(-1, 1) < 0 ? random(-3, -7) : random(3, 7);

    ball.left = random(strokeW + abs(ballXSpeed), cW - strokeW - ball.width - abs(ballXSpeed));
    ball.top = cH / 2 - strokeW / 2;

    DETECT_TRAECTORY_TRAIN_TEMP_STRUCT = {
      ballXSpeed,
      ballYSpeed,
      ballLeft: ball.left,
      ballTop: ball.top
    };
  }
}

function correctBallSpeedIfCollision() {
  const ballRect = {
    left: ball.left - abs(ballXSpeed),
    top: ball.top - abs(ballYSpeed),
    right: ball.left + ball.width + abs(ballXSpeed),
    bottom: ball.top + ball.height + abs(ballYSpeed)
  };

  const lWallRect = {
    left: lWall.left,
    top: lWall.top,
    right: lWall.left + lWall.width,
    bottom: lWall.top + lWall.height
  }

  const rWallRect = {
    left: rWall.left,
    top: rWall.top,
    right: rWall.left + rWall.width,
    bottom: rWall.top + rWall.height
  }

  const player2BarRect = {
    left: player2.left,
    top: player2.top,
    right: player2.left + player2.width,
    bottom: player2.top + player2.height
  }

  const player1BarRect = {
    left: player1.left,
    top: player1.top,
    right: player1.left + player1.width,
    bottom: player1.top + player1.height
  }

  const trainBarRect = {
    left: trainBar.left,
    top: trainBar.top,
    right: trainBar.left + trainBar.width,
    bottom: trainBar.top + trainBar.height
  }

  if (
    rectsIntersected(
      lWallRect,
      ballRect
    ) ||
    rectsIntersected(
      rWallRect,
      ballRect
    )
  ) {
    ballXSpeed = -ballXSpeed;
  }

  if (
    rectsIntersected(
      player2BarRect,
      ballRect
    ) ||
    (!collectDetectingData && rectsIntersected(
      player1BarRect,
      ballRect
    )) ||
    (collectDetectingData && rectsIntersected(
      trainBarRect,
      ballRect
    ))
  ) {
    ballYSpeed = -ballYSpeed;

    if (ball.top <= strokeW) {
      ball.top = strokeW + abs(ballYSpeed);
    }

    if (ball.top + ball.height >= cH - strokeW) {
      ball.top = cH - strokeW - ball.height - abs(ballYSpeed);
    }
  }
}

function rectsIntersected(r1, r2) {
  return !(r2.left > r1.right ||
    r2.right < r1.left ||
    r2.top > r1.bottom ||
    r2.bottom < r1.top);
}

function updateBallPosition() {
  ball.left = ball.left + ballXSpeed;
  ball.top = ball.top + ballYSpeed;
}

function updatePlayer1Position(model) {
  if (!model) return;

  const inputs = [ballXSpeed / MAX_ABS_SPEED, ballYSpeed / MAX_ABS_SPEED, ball.left / cW, ball.top / cH];

  const xs = tf.tensor2d([inputs])

  const ys = model.predict(xs);

  const resultNormalizedLeft = ys.dataSync();
  const left = resultNormalizedLeft * cW;

  if (player1.left > left) {
    updatePlayer1BarPosition(-1)
  }
  else {
    updatePlayer1BarPosition(1)
  }
}

function drawBall() {
  const yellow = '#ff0';

  fill(yellow);
  stroke(yellow);
  rect(ball.left, ball.top, ball.width, ball.height);
}

function drawPlayer1Bar() {
  const black = '#000';

  fill(black);
  stroke(black);
  rect(player1.left, player1.top, player1.width, player1.height);
}

function drawPlayer2Bar() {
  const black = '#000';

  fill(black);
  stroke(black);
  rect(player2.left, player2.top, player2.width, player2.height);
}

function drawTrainBar() {
  const red = '#f00';

  fill(red);
  stroke(red);
  rect(trainBar.left, trainBar.top, trainBar.width, trainBar.height);
}

function drawWalls() {
  const white = '#fff';

  fill(white);
  stroke(white);
  rect(lWall.left, lWall.top, lWall.width, lWall.height);

  fill(white);
  stroke(white);
  rect(rWall.left, rWall.top, rWall.width, rWall.height);
}
