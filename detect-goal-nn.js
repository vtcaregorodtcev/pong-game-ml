const tf = require('@tensorflow/tfjs-node');
const DETECT_GOAL_DATA = require('./detect-goal-data');

const MAX_ABS_SPEED = 5;
const LEARNING_RATE = 0.01;
const EPOCHS = 1000;
const BATCH_SIZE = 64;

function prepareData() {
  return DETECT_GOAL_DATA
    .map(item => {
      const { width, height } = item;

      const normalized = item.data.map(d => {
        const {
          ballXSpeed,
          ballYSpeed,
          ballLeft,
          ballTop,
          resultBallLeft
        } = d;

        return {
          // xspeed, yspeed, left, top
          inputs: [ballXSpeed / MAX_ABS_SPEED, ballYSpeed / MAX_ABS_SPEED, ballLeft / width, ballTop / height],
          // left
          target: [resultBallLeft / width]
        };
      });

      return normalized;
    })
    .flat();
}

const { xs, ys } = prepareData().reduce((acc, curr) => {
  acc.xs.push(curr.inputs);
  acc.ys.push(curr.target);

  return acc;
}, { xs: [], ys: [] });

// Instantiate the training tensors
let xTrain = tf.tensor2d(xs)
let yTrain = tf.tensor(ys.flat())

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [4], units: 64, activation: 'relu' }));
model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1 }));

model.compile({
  loss: 'meanSquaredError',
  optimizer: tf.train.rmsprop(LEARNING_RATE),
  metrics: ['mae', 'mse']
});

model.summary();

model.fit(xTrain, yTrain, {
  epochs: EPOCHS,
  batchSize: BATCH_SIZE,
  shuffle: true,
  validationData: [xTrain, yTrain]
})
  .then(() => model.save('file://./detect-goal-pp-v0'))
  .catch(e => console.log(e));
