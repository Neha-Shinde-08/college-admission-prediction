document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('predictionForm').addEventListener('submit', function (event) {
        event.preventDefault();
        fetch('/predict', {
            method: 'POST',
            body: new FormData(this),
        })
            .then(response => response.json())
            .then(data => {
                displayPredictions(data.result_xg, data.result_dt);
            })
            .catch(error => console.error('Error:', error));
    });
});

function displayPredictions(xgPredictions, dtPredictions) {
    const xgPredictionsContainer = document.getElementById('xg_predictions');
    const dtPredictionsContainer = document.getElementById('dt_predictions');

    xgPredictionsContainer.innerHTML = '<h2>XGBoost Predicted Colleges:</h2>';
    dtPredictionsContainer.innerHTML = '<h2>Decision Tree Predicted Colleges:</h2>';

    xgPredictionsContainer.innerHTML += '<table>';
    dtPredictionsContainer.innerHTML += '<table>';

    xgPredictionsContainer.innerHTML += '<tr><th>Rank</th><th>College</th></tr>';
    dtPredictionsContainer.innerHTML += '<tr><th>Rank</th><th>College</th></tr>';

    xgPredictions.forEach((college, index) => {
        xgPredictionsContainer.innerHTML += `<tr><td>${index + 1}</td><td>${college}</td></tr>`;
    });

    dtPredictions.forEach((college, index) => {
        dtPredictionsContainer.innerHTML += `<tr><td>${index + 1}</td><td>${college}</td></tr>`;
    });

    xgPredictionsContainer.innerHTML += '</table>';
    dtPredictionsContainer.innerHTML += '</table>';
}
