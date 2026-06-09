// =========================
// APP START
// =========================


let currentDataset = null;
let currentModel = null;
let modelConfig = {
    layers: 2,
    n1: 100,
    n2: 100,
    n3: 100,
    activation: "relu"
};

window.addEventListener("load", initApp);

function logRun(stage, dataset) {

    const cfg = modelConfig;

    console.log(
        `[${stage}]`,
        {
            dataPoints: dataset?.xTrain?.length ?? "unknown",
            layers: cfg.layers,
            neurons: {
                n1: cfg.n1,
                n2: cfg.n2,
                n3: cfg.n3
            }
        }
    );
}

/* =========================
   LOADING STATES
========================= */

function showLoading(id, text = "Loading...") {

    const el = document.getElementById(id);

    if (!el) return;

    el.innerHTML = `
        <div class="loading-box">
            <div class="spinner"></div>
            <p>${text}</p>
        </div>
    `;
}

function clearPlot(id) {

    const el = document.getElementById(id);

    if (!el) return;

    el.innerHTML = "";
}

async function initApp() {
    const settings = loadSettings(false);
    // UI mit gespeicherten Werten aktualisieren
    document.getElementById("numPoints").value = settings.N;
    document.getElementById("noiseVar").value = settings.noiseVar;

    document.getElementById("numPointsVal").textContent = settings.N;
    document.getElementById("noiseVarVal").textContent = settings.noiseVar;

    // Slider aktualisieren
    registerSliderUpdates();

    // Initiale Daten erzeugen
    generateDatasetAndTrain();
}

function loadSettings(useSaved = false) {

    if (!useSaved) {
        return {
            N: 100,
            noiseVar: 0.05
        };
    }

    return {
        N: parseInt(localStorage.getItem("param_N")) || 100,
        noiseVar: parseFloat(localStorage.getItem("param_noiseVar")) || 0.05
    };
}

function generateDataset(
    N = 100,
    noiseVar = 0.05,
    seed = 42
) {

    // Reproduzierbar
    if (typeof Math.seedrandom === "function") {
        Math.seedrandom(seed);
    }

    // Ziel-Funktion
    const targetFunction = x =>

        0.5 *
        (x + 0.8) *
        (x + 1.8) *
        (x - 0.2) *
        (x - 0.3) *
        (x - 1.9) + 1;

    // Gaußsches Rauschen
    function gaussianNoise(
        mean = 0,
        stdDev = 1
    ) {

        let u = 0;
        let v = 0;

        while (u === 0) {
            u = Math.random();
        }

        while (v === 0) {
            v = Math.random();
        }

        return (
            stdDev *
            Math.sqrt(-2 * Math.log(u)) *
            Math.cos(2 * Math.PI * v) +
            mean
        );
    }

    // x-Werte
    const xData = Array.from(
        { length: N },
        () => Math.random() * 4 - 2
    );

    // y-Werte
    const yClean = xData.map(targetFunction);

    const yNoisy = yClean.map(
        y => y + gaussianNoise(
            0,
            Math.sqrt(noiseVar)
        )
    );

    // Zufällige Indizes
    const indices = Array.from(
        { length: N },
        (_, i) => i
    );

    // Shuffle
    indices.sort(() => Math.random() - 0.5);
  //  tf.util.shuffle(indices);

    // 50% / 50%
    const splitIndex = Math.floor(N / 2);

    const trainIndices =
        indices.slice(0, splitIndex);

    const testIndices =
        indices.slice(splitIndex);

    // Hilfsfunktion
    const pick = (arr, idx) =>
        idx.map(i => arr[i]);

    return {

        // Unverrauscht
        xTrain: pick(xData, trainIndices),
        yTrainClean: pick(yClean, trainIndices),

        xTest: pick(xData, testIndices),
        yTestClean: pick(yClean, testIndices),

        // Verrauscht
        yTrainNoisy: pick(yNoisy, trainIndices),
        yTestNoisy: pick(yNoisy, testIndices)
    };
}

function saveDataset() {

    if (!currentDataset) return;

    const payload = {
        dataset: currentDataset,
        config: {
            N: currentDataset.xTrain.length,
            noiseVar: parseFloat(document.getElementById("noiseVar").value)
        }
    };

    localStorage.setItem("dataset", JSON.stringify(payload));
}

function toTensor(x, y) {

    return [

        tf.tensor2d(x, [x.length, 1]),
        tf.tensor2d(y, [y.length, 1])
    ];
}

function createModel() {

    const model = tf.sequential();

    const cfg = modelConfig;

    const units = [cfg.n1, cfg.n2, cfg.n3];

    // reproduzierbare Initialisierung
    const seed = 42;

    // Hidden Layers
    for (let i = 0; i < cfg.layers; i++) {

        model.add(tf.layers.dense({

            units: units[i],

            activation: cfg.activation,

            inputShape: i === 0 ? [1] : undefined,

            // =========================
            // INITIALIZER AUS CODE 2
            // =========================

            kernelInitializer:
                tf.initializers.glorotUniform({
                    seed
                }),

            biasInitializer:
                tf.initializers.zeros()
        }));
    }

    // Output Layer
    model.add(tf.layers.dense({

        units: 1,

        kernelInitializer:
            tf.initializers.glorotUniform({
                seed
            }),

        biasInitializer:
            tf.initializers.zeros()
    }));

    model.compile({

        optimizer: tf.train.adam(0.01),

        loss: "meanSquaredError"
    });

    return model;
}

async function evaluateModel(model, xTensor, yTensor) {

    const predictions = model.predict(xTensor);

    const mseTensor = tf.metrics.meanSquaredError(
        yTensor,
        predictions
    );

    const mse =
        (await mseTensor.mean().data())[0];

    const predArray =
        (await predictions.array())
            .map(v => v[0]);

    return {
        mse,
        predictions: predArray
    };
}

function generateDatasetAndTrain() {

    // Alte Plots löschen
    document.getElementById("r1_clean").innerHTML = "";
    document.getElementById("r1_noisy").innerHTML = "";

    // Werte aus UI lesen
    const N = parseInt(
        document.getElementById("numPoints").value
    );

    const noiseVar = parseFloat(
        document.getElementById("noiseVar").value
    );

    // Einstellungen speichern
    localStorage.setItem("param_N", N);
    localStorage.setItem("param_noiseVar", noiseVar);

    // Datensatz erzeugen
    const dataset = generateDataset(N, noiseVar);
    currentDataset = dataset;

    // Unverrauschte Daten
    drawDatasetChart(
        "r1_clean",

        dataset.xTrain,
        dataset.yTrainClean,

        dataset.xTest,
        dataset.yTestClean,

        "Unverrauschte Daten"
    );

    // Verrauschte Daten
    drawDatasetChart(
        "r1_noisy",

        dataset.xTrain,
        dataset.yTrainNoisy,

        dataset.xTest,
        dataset.yTestNoisy,

        "Verrauschte Daten"
    );

    trainCleanModel().then(async() => {
        currentModel = createModel();
        const model = currentModel;
        await trainR3BestFit();
        await trainA4Overfit();
    });
}

async function applyArchitectureChange() {
console.log("applyArchitectureChange");
    updateArchitecture();

    // UI reset
    clearPlot("r2_train");
    clearPlot("r2_test");
    clearPlot("r3_train");
    clearPlot("r3_test");
    clearPlot("r4_train");
    clearPlot("r4_test");

    await tf.nextFrame();

    // neu trainieren
    await trainCleanModel();
    await trainR3BestFit();
    await trainA4Overfit();
}

async function trainCleanModel() {
console.log('trainCleanModel beginnt');
    // Alte R2 Plots löschen
    document.getElementById("r2_train").innerHTML = "";
    document.getElementById("r2_test").innerHTML = "";

    const dataset = currentDataset;
    logRun("R2", dataset);

    showLoading("r2_train", "Trainingsdaten werden geladen...");
    showLoading("r2_test", "Testdaten werden geladen...");

    await tf.nextFrame();

    // Tensoren erzeugen
    const [xTrainT, yTrainT] = toTensor(
        dataset.xTrain,
        dataset.yTrainClean
    );

    const [xTestT, yTestT] = toTensor(
        dataset.xTest,
        dataset.yTestClean
    );

    currentModel = createModel();
    const model = currentModel;

    // Training
    await model.fit(

        xTrainT,
        yTrainT,

        {
            epochs: 1000,
            shuffle: false
        }
    );

    // Evaluation
    const trainEval =
        await evaluateModel(
            model,
            xTrainT,
            yTrainT
        );

    const testEval =
        await evaluateModel(
            model,
            xTestT,
            yTestT
        );

    clearPlot("r2_train");
    clearPlot("r2_test");

    // Training Plot
    drawPredictionChart(

        "r2_train",

        dataset.xTrain,
        dataset.yTrainClean,

        trainEval.predictions,

        "Training",
        trainEval.mse,
        { showLine: false }
    );

    // Test Plot
    drawPredictionChart(

        "r2_test",

        dataset.xTest,
        dataset.yTestClean,

        testEval.predictions,

        "Test",
        testEval.mse,
        { showLine: false }
    );

    // Loss Text
    document.getElementById(
        "r2_loss_text"
    ).innerHTML = `

        <b>Train Loss:</b>
        ${trainEval.mse.toFixed(6)}

        <br>

        <b>Test Loss:</b>
        ${testEval.mse.toFixed(6)}
    `;
}


// =========================
// CHART
// =========================

function drawDatasetChart(

    containerId,

    xTrain,
    yTrain,

    xTest,
    yTest,

    title
) {

    const canvas =
        document.createElement("canvas");

    document
        .getElementById(containerId)
        .appendChild(canvas);

    new Chart(canvas, {

        type: "scatter",

        data: {

            datasets: [

                {
                    label: "Training",

                    data: xTrain.map(
                        (x, i) => ({
                            x,
                            y: yTrain[i]
                        })
                    ),

                    backgroundColor: "#3498db",
                    pointRadius: 3
                },

                {
                    label: "Test",

                    data: xTest.map(
                        (x, i) => ({
                            x,
                            y: yTest[i]
                        })
                    ),

                    backgroundColor: "#e67e22",
                    pointRadius: 3
                }
            ]
        },

        options: {

            responsive: true,
            maintainAspectRatio: false,
            plugins: {

                title: {
                    display: true,
                    text: title
                },

                legend: {
                    display: true
                }
            },

            scales: {

                x: {
                    title: {
                        display: true,
                        text: "x"
                    }
                },

                y: {
                    title: {
                        display: true,
                        text: "y"
                    }
                }
            }
        }
    });
}


function drawPredictionChart(
    containerId,
    x,
    yTrue,
    yPred,
    title,
    mse,
    options = {}
) {
    const {
        showLine = true,
        pointRadius = 3
    } = options;

    const canvas = document.createElement("canvas");
    document.getElementById(containerId).appendChild(canvas);

    const combined = x.map((val, i) => ({
        x: val,
        yTrue: yTrue[i],
        yPred: yPred[i]
    }));

    combined.sort((a, b) => a.x - b.x);

    new Chart(canvas, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Echte Werte",
                    data: combined.map(p => ({ x: p.x, y: p.yTrue })),
                    backgroundColor: "#3498db",
                    pointRadius: 2
                },
                {
                    label: "Vorhersage",
                    data: combined.map(p => ({ x: p.x, y: p.yPred })),
                    type: showLine ? "line" : "scatter",
                    borderColor: "#e74c3c",
                    backgroundColor: "#e74c3c",
                    fill: false,
                    pointRadius: 3,
                    showLine: false,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `${title} | MSE: ${mse.toFixed(6)}`
                }
            },
            scales: {
                x: { title: { display: true, text: "x" } },
                y: { title: { display: true, text: "y" } }
            }
        }
    });
}

// =========================
// UI
// =========================

function updateArchitecture() {
console.log("Architektur")
    const layers = parseInt(
        document.getElementById("hiddenLayers")?.value || 2
    );

    const n1 = parseInt(document.getElementById("n1")?.value || 100);
    const n2 = parseInt(document.getElementById("n2")?.value || 100);
    const n3 = parseInt(document.getElementById("n3")?.value || 100);

    modelConfig = {
        layers,
        n1,
        n2,
        n3,
        activation: "relu"
    };

    // optional UI feedback
    console.log("Architektur aktualisiert:", modelConfig);
}

function registerSliderUpdates() {


    //const $ = id => document.getElementById(id);
    // Datenpunkte
    document.getElementById("numPoints").addEventListener("input", e => {

        document.getElementById("numPointsVal").textContent = e.target.value;
    });

    // Noise
    document.getElementById("noiseVar")
        .addEventListener("input", e => {

            document.getElementById("noiseVarVal").textContent = e.target.value;

        });

    // Hidden Layers
    document.getElementById("hiddenLayers").addEventListener("input", e => {
        document.getElementById("hiddenLayersVal").textContent = e.target.value;
        updateArchitecture();
    });

    ["n1", "n2", "n3"].forEach(id => {

        const el = document.getElementById(id);
        const label = document.getElementById(id + "Val");

        if (!el || !label) return;

        el.addEventListener("input", e => {
            label.textContent = e.target.value;
            updateArchitecture();
        });
    });

    // R3 Epochs
    const r3Slider = document.getElementById("r3_epochs");
    const r3Label = document.getElementById("r3_epochsVal");

    if (r3Slider && r3Label) {

        r3Slider.addEventListener("input", e => {
            r3Label.textContent = e.target.value;
        });
        r3Label.textContent = r3Slider.value;
    }
    const r3Btn = document.getElementById("r3_retrain");

    if (r3Btn) {

        r3Btn.addEventListener("click", async () => {

            r3Btn.disabled = true;

            document.getElementById("r3_train").innerHTML = "";
            document.getElementById("r3_test").innerHTML = "";
            document.getElementById("r3_loss_chart").innerHTML = "";

            await tf.nextFrame();

            await trainR3BestFit();

            r3Btn.disabled = false;
        });
    }

    // R4 Epochs
    const r4Slider = document.getElementById("epochs");
    const r4Label = document.getElementById("r4_epochs");

    if (r4Slider && r4Label) {

        // Label initial synchronisieren
        r4Label.textContent = r4Slider.value;

        // Live Update beim Sliden
        r4Slider.addEventListener("input", e => {
            r4Label.textContent = e.target.value;
        });


    }
}

function redrawAllFromDataset() {

    const dataset = currentDataset;

    // R1
    clearPlot("r1_clean");
    clearPlot("r1_noisy");

    drawDatasetChart(
        "r1_clean",
        dataset.xTrain,
        dataset.yTrainClean,
        dataset.xTest,
        dataset.yTestClean,
        "Unverrauschte Daten"
    );

    drawDatasetChart(
        "r1_noisy",
        dataset.xTrain,
        dataset.yTrainNoisy,
        dataset.xTest,
        dataset.yTestNoisy,
        "Verrauschte Daten"
    );

    // danach neu trainieren (wichtig für R2–R4)
    trainCleanModel().then(async () => {
        await trainR3BestFit();
        await trainA4Overfit();
    });
}


function loadDataset() {

    const data = localStorage.getItem("dataset");

    if (!data) {
        console.warn("Kein gespeichertes Dataset gefunden");
        return;
    }

    let parsed;
    try {
        parsed = JSON.parse(data);
    } catch (e) {
        console.error("Dataset beschädigt:", e);
        return;
    }

    console.log("Dataset geladen:", parsed);

    currentDataset = parsed.dataset ?? parsed; // wichtig wegen deiner Save-Struktur

    // =========================
    // UI synchronisieren
    // =========================

    const N = currentDataset.xTrain.length;

    document.getElementById("numPoints").value = N;
    document.getElementById("numPointsVal").textContent = N;

    const noiseVar =
        parsed.config?.noiseVar ??
        parseFloat(localStorage.getItem("param_noiseVar")) ??
        0.05;

    document.getElementById("noiseVar").value = noiseVar;
    document.getElementById("noiseVarVal").textContent = noiseVar;

    // =========================
    // Visualisierung
    // =========================
    redrawAllFromDataset();
}

async function saveModel() {

    if (!currentModel) return;

    const config = modelConfig;

    // Modell speichern
    await currentModel.save("localstorage://ffnn-regression-model");

    // Config separat speichern
    localStorage.setItem(
        "ffnn-model-config",
        JSON.stringify(config)
    );

    console.log("Model + Config gespeichert ✔");
}

async function loadModel() {

    try {
        // 1. Modell laden
        const model = await tf.loadLayersModel(
            "localstorage://ffnn-regression-model"
        );

        currentModel = model;

        // 2. Config laden
        const configRaw = localStorage.getItem("ffnn-model-config");

        if (!configRaw) {
            console.warn("Keine Config gefunden");
            return;
        }

        const config = JSON.parse(configRaw);

        modelConfig = config;

        console.log("Model + Config geladen ✔", config);

        // Slider setzen
        document.getElementById("hiddenLayers").value = config.layers;
        document.getElementById("n1").value = config.n1;
        document.getElementById("n2").value = config.n2;
        document.getElementById("n3").value = config.n3;

        // Labels aktualisieren
        document.getElementById("hiddenLayersVal").textContent = config.layers;
        document.getElementById("n1Val").textContent = config.n1;
        document.getElementById("n2Val").textContent = config.n2;
        document.getElementById("n3Val").textContent = config.n3;

        // Aktivierung (falls vorhanden)
        if (config.activation) {
            document.getElementById("activation").value = config.activation;
        }

        // Wichtig: interne Architektur aktualisieren
        updateArchitecture();

        console.log("UI synchronisiert ✔");
    } catch (err) {
        console.error("Load error:", err);
    }
}

async function trainR3BestFit() {

    const dataset = currentDataset;
    if (!dataset) return;

    logRun("R3", dataset);

    document.getElementById("r3_train").innerHTML = "";
    document.getElementById("r3_test").innerHTML = "";
    document.getElementById("r3_loss_chart").innerHTML = "";

    showLoading("r3_train", "Training...");
    showLoading("r3_test", "Test...");
    showLoading("r3_loss_chart", "Training läuft...");

    await tf.nextFrame();

    const epochs = parseInt(document.getElementById("r3_epochs").value);

    currentModel = createModel();
    const model = currentModel;

    const [xTrainT, yTrainT] = toTensor(dataset.xTrain, dataset.yTrainNoisy);
    const [xTestT, yTestT] = toTensor(dataset.xTest, dataset.yTestNoisy);

    // =========================
    // Loss SPEICHERN (nicht plotten!)
    // =========================
    const trainLoss = [];
    const testLoss = [];

    // =========================
    // TRAINING (ohne Chart)
    // =========================
    for (let epoch = 1; epoch <= epochs; epoch++) {

        await model.fit(xTrainT, yTrainT, {
            epochs: 1,
            shuffle: false
        });

        const trainEval = await evaluateModel(model, xTrainT, yTrainT);
        const testEval = await evaluateModel(model, xTestT, yTestT);

        trainLoss.push(trainEval.mse);
        testLoss.push(testEval.mse);
    }

    const container = document.getElementById("r3_loss_chart");
    container.innerHTML = "";

    const canvas = document.createElement("canvas");
    container.appendChild(canvas);

    new Chart(canvas, {
        type: "line",
        data: {
            labels: trainLoss.map((_, i) => i + 1),
            datasets: [
                {
                    label: "Train Loss",
                    data: trainLoss,
                    borderColor: "#3498db",
                    pointRadius: 0,
                    tension: 0
                },
                {
                    label: "Test Loss",
                    data: testLoss,
                    borderColor: "#e74c3c",
                    pointRadius: 0,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: "R3 Lernkurve (final)"
                }
            }
        }
    });

    // =========================
    // FINAL PREDICTIONS
    // =========================
    const trainEvalFinal = await evaluateModel(model, xTrainT, yTrainT);
    const testEvalFinal = await evaluateModel(model, xTestT, yTestT);

    clearPlot("r3_train");
    clearPlot("r3_test");

    drawPredictionChart(
        "r3_train",
        dataset.xTrain,
        dataset.yTrainNoisy,
        trainEvalFinal.predictions,
        "R3 Train",
        trainEvalFinal.mse,
        { showLine: false }
    );

    drawPredictionChart(
        "r3_test",
        dataset.xTest,
        dataset.yTestNoisy,
        testEvalFinal.predictions,
        "R3 Test",
        testEvalFinal.mse,
        { showLine: false }
    );
}

async function trainA4Overfit() {
    const dataset = currentDataset;
    if (!dataset) return;

    logRun("R4", dataset);

    // Reset Plots
    document.getElementById("r4_train").innerHTML = "";
    document.getElementById("r4_test").innerHTML = "";
    document.getElementById("r4_loss_chart").innerHTML = "";

    showLoading("r4_train", "Trainingsdaten werden geladen...");
    showLoading("r4_test", "Testdaten werden geladen...");
    showLoading("r4_loss_chart", "Loss Chart wird berechnet...");

    await tf.nextFrame();

    console.log("R4 START");
    console.log("Dataset:", dataset);
    console.log("r4_train:", document.getElementById("r4_train"));
    console.log("r4_loss_chart:", document.getElementById("r4_loss_chart"));


    const epochs = parseInt(
        document.getElementById("r4_epochs")?.value || 2000
    );

    const hiddenUnits = [100, 100];
    const activation = "relu";

    currentModel = createModel();
    const model = currentModel;


    const [xTrainT, yTrainT] = toTensor(
        dataset.xTrain,
        dataset.yTrainNoisy
    );


    const [xTestT, yTestT] = toTensor(
        dataset.xTest,
        dataset.yTestNoisy
    );

    // =========================
    // LOSS CHART (A4)
    // =========================

    clearPlot("r4_loss_chart");
    const canvas = document.createElement("canvas");
    document.getElementById("r4_loss_chart").appendChild(canvas);

    const lossChart = new Chart(canvas, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Train Loss",
                    data: [],
                    borderColor: "#3498db",
                    pointRadius: 0,
                    tension: 0
                },
                {
                    label: "Test Loss",
                    data: [],
                    borderColor: "#e74c3c",
                    pointRadius: 0,
                    tension: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: "A4 Overfitting: Train vs Test MSE"
                }
            },
            scales: {
                x: {
                    title: { display: true, text: "Epoche" }
                },
                y: {
                    title: { display: true, text: "Loss (MSE)" }
                }
            }
        }
    });

    // =========================
    // TRAINING LOOP
    // =========================
    for (let epoch = 1; epoch <= epochs; epoch++) {

        await model.fit(xTrainT, yTrainT, {
            epochs: 1,
            shuffle: false
        });

        //  Loss nur messen, nicht zum Steuern nutzen
        const trainEval = await evaluateModel(model, xTrainT, yTrainT);
        const testEval  = await evaluateModel(model, xTestT, yTestT);

        lossChart.data.labels.push(epoch);
        lossChart.data.datasets[0].data.push(trainEval.mse);
        lossChart.data.datasets[1].data.push(testEval.mse);
        lossChart.update();
    }

    // =========================
    // FINAL PLOTS
    // =========================
    const trainFinal = await evaluateModel(model, xTrainT, yTrainT);
    const testFinal  = await evaluateModel(model, xTestT, yTestT);

    clearPlot("r4_train");
    clearPlot("r4_test");

    drawPredictionChart(
        "r4_train",
        dataset.xTrain,
        dataset.yTrainNoisy,
        trainFinal.predictions,
        "A4 Train (Overfit)",
        trainFinal.mse,
        { showLine: false }
    );

    drawPredictionChart(
        "r4_test",
        dataset.xTest,
        dataset.yTestNoisy,
        testFinal.predictions,
        "A4 Test (Overfit)",
        testFinal.mse,
        { showLine: false }
    );

    document.getElementById("r4_loss_text").innerHTML = `
        <b>Train Loss:</b> ${trainFinal.mse.toFixed(6)}
        <br>
        <b>Test Loss:</b> ${testFinal.mse.toFixed(6)}
    `;
}