console.log('Hello TensorFlow');

let currentData = null;
let currentModel = null;
let modelConfig = {
    layers: 2,
    n1: 100,
    n2: 100,
    n3: 100,
    activation: 'relu'
};

/* =========================
   1. DATENGENERIERUNG
========================= */

function groundTruth(x) {
    return 0.5 *
        (x + 0.8) *
        (x + 1.8) *
        (x - 0.2) *
        (x - 0.3) *
        (x - 1.9) + 1;
}

function generateData(numPoints = 100) {

    const data = [];

    for (let i = 0; i < numPoints; i++) {
        const x = Math.random() * 4 - 2;
        const y = groundTruth(x);
        data.push({ x, y });
    }

    return data;
}




function splitData(data) {

    tf.util.shuffle(data);

    return {
        train: data.slice(0, 50),
        test: data.slice(50)
    };
}

function gaussianRandom(mean = 0, stdDev = 1) {

    let u = 0;
    let v = 0;

    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();

    return mean +
        stdDev *
        Math.sqrt(-2.0 * Math.log(u)) *
        Math.cos(2.0 * Math.PI * v);
}

function addNoise(data, variance = 0.05) {

    const stdDev = Math.sqrt(variance);

    return data.map(p => ({
        x: p.x,
        y: p.y + gaussianRandom(0, stdDev)
    }));
}


function createDataset(numPoints, noiseVar) {

    const clean = generateData(numPoints);
    const { train, test } = splitData(clean);

    return {
        train,
        test,
        noisyTrain: addNoise(train, noiseVar),
        noisyTest: addNoise(test, noiseVar)
    };
}

function generateDatasetAndTrain() {
    setTimeout(async () => {
        const numPoints = parseInt(document.getElementById("numPoints").value);
        const noiseVar = parseFloat(document.getElementById("noiseVar").value);

        currentData = createDataset(numPoints, noiseVar);

        console.log("Dataset erstellt:", currentData);

        renderR1(
            currentData.train,
            currentData.test,
            currentData.noisyTrain,
            currentData.noisyTest
        );
        await rerunAll();
    },50);

}

function saveDataset() {
    if (!currentData) {
        alert("Kein Datensatz vorhanden!");
        return;
    }

    const payload = {
        data: currentData,
        createdAt: new Date().toISOString(),
        version: "1.0"
    };

    localStorage.setItem("dataset", JSON.stringify(payload));

    console.log("Dataset gespeichert");
}

async function loadDataset() {
    const raw = localStorage.getItem("dataset");

    if (!raw) {
        alert("Kein gespeicherter Datensatz gefunden!");
        return;
    }

    const parsed = JSON.parse(raw);

    if (!parsed.data) {
        alert("Ungültiger Datensatz!");
        return;
    }

    currentData = parsed.data;

    console.log("Dataset geladen:", parsed);

    renderR1(
        currentData.train,
        currentData.test,
        currentData.noisyTrain,
        currentData.noisyTest
    );
    await rerunAll();

}

/* =========================
   2. MODEL
========================= */

function createModel() {

    const model = tf.sequential();

    const activation = modelConfig.activation;

    model.add(tf.layers.dense({
        inputShape: [1],
        units: modelConfig.n1,
        activation
    }));

    if (modelConfig.layers >= 2) {
        model.add(tf.layers.dense({
            units: modelConfig.n2,
            activation
        }));
    }

    if (modelConfig.layers === 3) {
        model.add(tf.layers.dense({
            units: modelConfig.n3,
            activation
        }));
    }

    model.add(tf.layers.dense({
        units: 1,
        activation: 'linear'
    }));

    return model;
}

async function trainModel(model, trainData, testData, epochs, name) {

    const xTrain = tf.tensor2d(trainData.map(d => d.x), [trainData.length, 1]);
    const yTrain = tf.tensor2d(trainData.map(d => d.y), [trainData.length, 1]);

    const xTest = tf.tensor2d(testData.map(d => d.x), [testData.length, 1]);
    const yTest = tf.tensor2d(testData.map(d => d.y), [testData.length, 1]);

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });

    const history = {
        loss: [],
        val_loss: []
    };

    await model.fit(xTrain, yTrain, {
        epochs,
        batchSize: 32,
        shuffle: true,
        validationData: [xTest, yTest],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {

                history.loss.push(logs.loss);
                history.val_loss.push(logs.val_loss);

                console.log(`${name} Epoch ${epoch}: loss=${logs.loss}`);
            }
        }
    });

    return history;
}

async function trainCurrentModel() {
    if (currentModel) {
        currentModel.dispose();
    }
    if (!currentData) {
        alert("Kein Dataset geladen!");
        return;
    }

    const epochs = parseInt(document.getElementById("epochs").value);

    currentModel = createModel();

    await trainModel(
        currentModel,
        currentData.noisyTrain,
        currentData.noisyTest,
        epochs,
        "Interactive Model"
    );

    console.log("Training fertig");
}


function testCurrentModel() {

    if (!currentModel || !currentData) return;

    const preds = predict(currentModel, currentData.noisyTest);

    renderPlot(
        "r4_test",
        [currentData.noisyTest, preds],
        "Interactive Test"
    );

    console.log("Model getestet");
}

/* =========================
   3. PREDICTION
========================= */

function predict(model, data) {

    const xs = tf.tensor2d(
        data.map(d => d.x),
        [data.length, 1]
    );

    const predsTensor = model.predict(xs);

    const preds = Array.from(predsTensor.dataSync()).map((y, i) => ({
        x: data[i].x,
        y
    }));

    xs.dispose();
    predsTensor.dispose();

    return preds;
}

async function saveModel() {
    if (!currentModel) return;

    await currentModel.save("localstorage://my-model");
    console.log("Model gespeichert");
}

async function loadModel() {
    currentModel = await tf.loadLayersModel("localstorage://my-model");
    console.log("Model geladen");
}

/* =========================
   4. VISUALISIERUNG
========================= */

function renderPlot(id, data, title) {

    tfvis.render.scatterplot(
        document.getElementById(id),
        {
            values: data,
            series: ['Train', 'Test']
        },
        {
            xLabel: 'x',
            yLabel: 'y',
            height: 300,
            title: title
        }
    );
}

function renderLossCurve(history, containerId, title) {

    tfvis.render.linechart(
        document.getElementById(containerId),
        {
            values: [
                history.loss.map((y, x) => ({ x, y })),
                history.val_loss.map((y, x) => ({ x, y }))
            ],
            series: ['Train Loss', 'Test Loss']
        },
        {
            xLabel: 'Epoch',
            yLabel: 'Loss (MSE)',
            height: 300
        }
    );
}

/* =========================
   5. R1–R4 LAYOUT
========================= */

function renderR1(cleanTrain, cleanTest, noisyTrain, noisyTest) {

    tfvis.render.scatterplot(
        document.getElementById('r1_clean'),
        {
            values: [
                cleanTrain,
                cleanTest
            ],
            series: ['Train', 'Test']
        },
        { height: 300 }
    );

    tfvis.render.scatterplot(
        document.getElementById('r1_noisy'),
        {
            values: [
                noisyTrain,
                noisyTest
            ],
            series: ['Train', 'Test']
        },
        { height: 300 }
    );
}

/* R2: Clean Model */
function renderR2(model, train, test) {

    const pTrain = predict(model, train);
    const pTest = predict(model, test);

    tfvis.render.scatterplot(
        document.getElementById('r2_train'),
        {
            values: [
                train,
                pTrain
            ],
            series: ['True', 'Prediction']
        },
        { height: 300 }
    );

    tfvis.render.scatterplot(
        document.getElementById('r2_test'),
        {
            values: [
                test,
                pTest
            ],
            series: ['True', 'Prediction']
        },
        { height: 300 }
    );
}
/* R3: Best Fit */
function renderR3(model, train, test) {

    const pTrain = predict(model, train);
    const pTest = predict(model, test);

    renderPlot('r3_train', [train, pTrain], 'Best Fit Train');
    renderPlot('r3_test', [test, pTest], 'Best Fit Test');
}

/* R4: Overfit */
function renderR4(model, train, test) {

    const pTrain = predict(model, train);
    const pTest = predict(model, test);

    renderPlot('r4_train', [train, pTrain], 'Overfit Train');
    renderPlot('r4_test', [test, pTest], 'Overfit Test');
}

/* =========================
   6. MAIN PIPELINE
========================= */

async function trainBestModel() {

    if (!currentData) {
        alert("Bitte zuerst Dataset generieren!");
        return;
    }

    const epochs = parseInt(document.getElementById("epochs").value);

    currentModel = createModel();

    const history = await trainModel(
        currentModel,
        currentData.noisyTrain,
        currentData.noisyTest,
        epochs,
        "Best Fit"
    );

    console.log("Training fertig");

    renderR3(currentModel, currentData.noisyTrain, currentData.noisyTest);

    renderLossCurve(history, 'r3_loss', 'Training Curve');
}

function updateArchitecture() {

    // 🔥 Altes Modell aus Speicher entfernen
    if (currentModel) {
        currentModel.dispose();
        currentModel = null;

        console.log("Altes Modell entfernt");
    }

    // Neue Architektur übernehmen
    modelConfig.layers = parseInt(
        document.getElementById("hiddenLayers").value
    );

    modelConfig.n1 = parseInt(
        document.getElementById("n1").value
    );

    modelConfig.n2 = parseInt(
        document.getElementById("n2").value
    );

    modelConfig.n3 = parseInt(
        document.getElementById("n3").value
    );

    modelConfig.activation =
        document.getElementById("activation").value;

    console.log("Neue Architektur:", modelConfig);
}

function renderLossText(id, trainLoss, testLoss) {

    document.getElementById(id).innerHTML = `
        <p>Train Loss: ${trainLoss.toFixed(6)}</p>
        <p>Test Loss: ${testLoss.toFixed(6)}</p>
    `;
}

async function initApp() {

    console.log("App startet mit Standardwerten");

    const numPoints = 100;
    const noiseVar = 0.05;

    currentData = createDataset(numPoints, noiseVar);

    renderR1(
        currentData.train,
        currentData.test,
        currentData.noisyTrain,
        currentData.noisyTest
    );

    // =========================
    // R2
    // =========================
    const cleanModel = createModel();

    const cleanHistory = await trainModel(
        cleanModel,
        currentData.train,
        currentData.test,
        100,
        "Clean Model"
    );

    const r2TrainLoss = cleanHistory.loss.at(-1);
    const r2TestLoss = cleanHistory.val_loss.at(-1);

    console.log("R2 Train Loss:", r2TrainLoss);
    console.log("R2 Test Loss:", r2TestLoss);

    renderLossText(
        "r2_loss_text",
        r2TrainLoss,
        r2TestLoss
    );

//    const predTrain = predict(cleanModel, currentData.train);
//    const predTest = predict(cleanModel, currentData.test);

    renderR2(cleanModel, currentData.train, currentData.test);

//    tfvis.render.scatterplot(
//        document.getElementById('r2_train'),
//        { values: [currentData.train, predTrain], series: ['True', 'Pred'] },
//        { height: 300 }
//    );

//    tfvis.render.scatterplot(
//        document.getElementById('r2_test'),
//        { values: [currentData.test, predTest], series: ['True', 'Pred'] },
//        { height: 300 }
//    );

    // =========================
    // R3
    // =========================
    const bestModel = createModel();

    const bestHistory = await trainModel(
        bestModel,
        currentData.noisyTrain,
        currentData.noisyTest,
        120,
        "Best Model"
    );

    const r3TrainLoss = bestHistory.loss.at(-1);
    const r3TestLoss = bestHistory.val_loss.at(-1);

    console.log("R3 Train Loss:", r3TrainLoss);
    console.log("R3 Test Loss:", r3TestLoss);

    renderLossText(
        "r3_loss_text",
        r3TrainLoss,
        r3TestLoss
    );

    renderR3(bestModel, currentData.noisyTrain, currentData.noisyTest);
    renderLossCurve(bestHistory, 'r3_loss', 'R3 Loss');

    // =========================
    // R4
    // =========================
    const overfitModel = createModel();

    const overfitHistory = await trainModel(
        overfitModel,
        currentData.noisyTrain,
        currentData.noisyTest,
        2000,
        "Overfit Model"
    );

    const r4TrainLoss = overfitHistory.loss.at(-1);
    const r4TestLoss = overfitHistory.val_loss.at(-1);

    console.log("R4 Train Loss:", r4TrainLoss);
    console.log("R4 Test Loss:", r4TestLoss);

    renderR4(overfitModel, currentData.noisyTrain, currentData.noisyTest);

    console.log("INIT DONE");
}

function clearR3Plots() {

    const ids = [
        "r3_train",
        "r3_test",
        "r3_loss"
    ];

    ids.forEach(id => {

        const el = document.getElementById(id);

        if (!el) return;

        el.innerHTML = `
            <div class="loading-box">
                Training läuft...
            </div>
        `;
    });

    document.getElementById("r3_loss_text").innerHTML = "";
}

async function trainR3BestFit() {

    if (!currentData) {
        alert("Bitte zuerst Dataset generieren!");
        return;
    }

    const epochs = parseInt(
        document.getElementById("r3_epochs").value
    );

    console.log("R3 Training startet:", epochs);
    clearR3Plots();
    await new Promise(resolve => setTimeout(resolve, 50));

    const bestModel = createModel();

    const history = await trainModel(
        bestModel,
        currentData.noisyTrain,
        currentData.noisyTest,
        epochs,
        "Best Fit"
    );

    const trainLoss = history.loss.at(-1);
    const testLoss = history.val_loss.at(-1);

    renderR3(
        bestModel,
        currentData.noisyTrain,
        currentData.noisyTest
    );

    renderLossCurve(
        history,
        "r3_loss",
        "Best-Fit Loss Curve"
    );

    document.getElementById("r3_loss_text").innerHTML = `
        <p>Train Loss: ${trainLoss.toFixed(6)}</p>
        <p>Test Loss: ${testLoss.toFixed(6)}</p>
    `;
}

async function rerunAll() {

    if (!currentData) return;

    console.log("Re-running full pipeline...");

    /* =========================
       R2 CLEAN
    ========================= */
    const cleanModel = createModel();

    const cleanHistory = await trainModel(
        cleanModel,
        currentData.train,
        currentData.test,
        100,
        "Clean Model"
    );

    renderR2(cleanModel, currentData.train, currentData.test);

    renderLossText(
        "r2_loss_text",
        cleanHistory.loss.at(-1),
        cleanHistory.val_loss.at(-1)
    );

    /* =========================
       R3 BEST FIT
    ========================= */
    const bestModel = createModel();

    const bestHistory = await trainModel(
        bestModel,
        currentData.noisyTrain,
        currentData.noisyTest,
        120,
        "Best Model"
    );

    renderR3(bestModel, currentData.noisyTrain, currentData.noisyTest);

    renderLossCurve(bestHistory, "r3_loss", "R3 Loss");

    renderLossText(
        "r3_loss_text",
        bestHistory.loss.at(-1),
        bestHistory.val_loss.at(-1)
    );


    /* =========================
       R4 OVERFIT
    ========================= */
    const overfitModel = createModel();

    const overfitHistory = await trainModel(
        overfitModel,
        currentData.noisyTrain,
        currentData.noisyTest,
        200,
        "Overfit Model"
    );

    renderR4(overfitModel, currentData.noisyTrain, currentData.noisyTest);

    renderLossText(
        "r4_loss_text",
        overfitHistory.loss.at(-1),
        overfitHistory.val_loss.at(-1)
    );
    cleanModel.dispose();
    bestModel.dispose();
    overfitModel.dispose();
    console.log("Pipeline finished");
}


document.addEventListener("DOMContentLoaded", () => {

    initApp();

    function bind(id) {
        const el = document.getElementById(id);
        const out = document.getElementById(id + "Val");

        if (!el || !out) return;

        el.oninput = () => out.innerText = el.value;
    }

    [
        "numPoints",
        "noiseVar",
        "hiddenLayers",
        "n1",
        "n2",
        "n3",
        "r3_epochs"
    ]
        .forEach(bind);
});


window.generateDatasetAndTrain = generateDatasetAndTrain;
window.trainCurrentModel = trainCurrentModel;
window.testCurrentModel = testCurrentModel;
window.saveDataset = saveDataset;
window.loadDataset = loadDataset;
window.saveModel = saveModel;
window.loadModel = loadModel;
window.updateArchitecture = updateArchitecture;
window.trainR3BestFit = trainR3BestFit;