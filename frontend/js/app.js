const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");

let selectedFile = null;

imageInput.addEventListener("change", function () {
    selectedFile = this.files[0];

    if (!selectedFile) {
        fileName.textContent = "No image selected";
        return;
    }

    fileName.textContent = selectedFile.name;

    // Remove an existing analyze button
    const oldButton = document.getElementById("analyzeButton");
    if (oldButton) {
        oldButton.remove();
    }

    // Create Analyze button
    const button = document.createElement("button");
    button.id = "analyzeButton";
    button.className = "analyze-btn";
    button.textContent = "Analyze Road →";

    button.style.marginTop = "20px";
    button.style.padding = "14px 28px";
    button.style.border = "none";
    button.style.borderRadius = "10px";
    button.style.cursor = "pointer";
    button.style.fontSize = "16px";
    button.style.fontWeight = "600";

    button.onclick = analyzeRoad;

    imageInput.parentElement.appendChild(button);
});


async function analyzeRoad() {

    if (!selectedFile) {
        alert("Please select a road image first.");
        return;
    }

    const button = document.getElementById("analyzeButton");

    button.disabled = true;
    button.textContent = "Analyzing road...";

    // Remove previous results
    const oldResults = document.getElementById("results");
    if (oldResults) {
        oldResults.remove();
    }

    const formData = new FormData();

    formData.append("file", selectedFile);

    // Approximate camera horizontal FOV
    formData.append("hfov", "70");

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/analyze",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Analysis failed"
            );
        }

        displayResults(data);

    } catch (error) {

        console.error(error);

        alert(
            "Could not analyze the image.\n\n" +
            error.message
        );

    } finally {

        button.disabled = false;
        button.textContent = "Analyze Road →";
    }
}


function displayResults(data) {

    const results = document.createElement("section");

    results.id = "results";

    results.style.marginTop = "50px";
    results.style.padding = "30px";
    results.style.borderRadius = "20px";

    results.innerHTML = `
        <h2>Road Analysis Results</h2>

        <div style="
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
            gap:20px;
            margin-top:25px;
        ">

            <div class="result-card">
                <h3>Estimated Width</h3>
                <p style="font-size:32px;font-weight:700;">
                    ${data.estimated_width_m} m
                </p>
            </div>

            <div class="result-card">
                <h3>Estimated Range</h3>
                <p style="font-size:24px;font-weight:700;">
                    ${data.range_lower_m} –
                    ${data.range_upper_m} m
                </p>
            </div>

            <div class="result-card">
                <h3>Confidence</h3>
                <p style="font-size:32px;font-weight:700;">
                    ${data.overall_confidence}%
                </p>
            </div>

            <div class="result-card">
                <h3>Boundary Consistency</h3>
                <p style="font-size:32px;font-weight:700;">
                    ${data.consistency_percent}%
                </p>
            </div>

        </div>

        <div style="margin-top:30px;">

            <h3>Cross-Section Measurements</h3>

            <p>
                ${data.cross_sections
                    .map((width, index) =>
                        `Section ${index + 1}: ${width} m`
                    )
                    .join("<br>")
                }
            </p>

        </div>

        <div style="
            margin-top:30px;
            padding:20px;
            border-radius:12px;
            background:rgba(255,180,0,0.1);
        ">

            <strong>⚠ Measurement Notice</strong>

            <p>
                ${data.warning}
            </p>

        </div>
    `;

    document.body.appendChild(results);

    results.scrollIntoView({
        behavior: "smooth"
    });
}



// ============================================================
// HIGHWAY INFORMATION
// ============================================================

const highwayInput = document.getElementById("highwayInput");
const highwaySearchBtn = document.getElementById("highwaySearchBtn");
const highwayResult = document.getElementById("highwayResult");


// Search button
if (highwaySearchBtn) {

    highwaySearchBtn.addEventListener("click", function () {

        const highwayName = highwayInput.value.trim();

        if (!highwayName) {

            highwayResult.innerHTML = `
                <p>Please enter a highway name.</p>
            `;

            return;
        }

        searchHighway(highwayName);
    });
}


// Allow pressing Enter
if (highwayInput) {

    highwayInput.addEventListener("keydown", function (event) {

        if (event.key === "Enter") {

            searchHighway(highwayInput.value.trim());

        }

    });
}


// Search highway
async function searchHighway(highwayName) {

    if (!highwayName) {
        return;
    }

    highwayResult.innerHTML = `
        <p>Loading highway information...</p>
    `;

    try {

        const encodedName = encodeURIComponent(
            highwayName.toUpperCase()
        );

        const response = await fetch(
            `http://127.0.0.1:8000/road-info/${encodedName}`
        );

        if (!response.ok) {

            throw new Error(
                "Server returned an error."
            );

        }

        const result = await response.json();


        if (!result.found) {

            highwayResult.innerHTML = `
                <div class="highway-error">

                    <h3>Highway not found</h3>

                    <p>
                        Please search for one of:
                        NH 44, NH 48, NH 16, NH 27,
                        NH 19, NH 66 or NH 65.
                    </p>

                </div>
            `;

            return;
        }


        const data = result.data;


        highwayResult.innerHTML = `

            <div class="highway-header">

                <div>

                    <div class="badge">
                        NATIONAL HIGHWAY
                    </div>

                    <h2>
                        ${data.name}
                    </h2>

                </div>

            </div>


            <div class="highway-stats">

                <div class="highway-stat">

                    <span>Length</span>

                    <strong>
                        ${data.length}
                    </strong>

                </div>


                <div class="highway-stat">

                    <span>Starting Point</span>

                    <strong>
                        ${data.start}
                    </strong>

                </div>


                <div class="highway-stat">

                    <span>Ending Point</span>

                    <strong>
                        ${data.end}
                    </strong>

                </div>

            </div>


            <div class="highway-details">

                <div>

                    <h3>
                        States
                    </h3>

                    <div class="tag-list">

                        ${data.states.map(
                            state => `
                                <span class="road-tag">
                                    ${state}
                                </span>
                            `
                        ).join("")}

                    </div>

                </div>


                <div>

                    <h3>
                        Major Cities
                    </h3>

                    <div class="tag-list">

                        ${data.major_cities.map(
                            city => `
                                <span class="road-tag">
                                    ${city}
                                </span>
                            `
                        ).join("")}

                    </div>

                </div>

            </div>


            <div class="highway-description">

                <h3>
                    About this Highway
                </h3>

                <p>
                    ${data.description}
                </p>

                <p>
                    <strong>Importance:</strong>
                    ${data.importance}
                </p>

            </div>

        `;

    }

    catch (error) {

        console.error(
            "Highway search error:",
            error
        );

        highwayResult.innerHTML = `

            <div class="highway-error">

                <h3>
                    Unable to load highway information
                </h3>

                <p>
                    Make sure the Read the Road backend
                    is running.
                </p>

            </div>

        `;

    }
}