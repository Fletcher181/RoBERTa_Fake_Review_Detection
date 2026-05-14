document.getElementById("check").addEventListener("click", async () => {

    console.log("Button clicked");

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {

        chrome.tabs.sendMessage(
            tabs[0].id,
            { action: "GET_SELECTED_TEXT" },
            async (response) => {

                const reviewText = response.text;

                if (!reviewText) {
                    document.getElementById("result").innerText =
                        "Please highlight a review first.";
                    return;
                }

                const res = await fetch("http://127.0.0.1:8000/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ text: reviewText })
                });

                const data = await res.json();

                document.getElementById("result").innerText =
                    "Prediction: " + data.label;
            }
        );
    });
});