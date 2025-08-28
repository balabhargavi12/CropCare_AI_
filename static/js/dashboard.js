// Dashboard functionality
document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const resultsSection = document.getElementById('results-section');
    const previewImage = document.getElementById('preview-image');

    // Setup drag and drop
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Upload and get prediction
        uploadImage(file);
    }

    async function uploadImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                displayResults(data);
            } else {
                alert(data.message || 'Prediction failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('An error occurred during image analysis');
        }
    }

    function displayResults(data) {
        // Show results section
        resultsSection.style.display = 'block';

        // Update results
        document.getElementById('disease-result').textContent = data.prediction;
        document.getElementById('confidence-result').textContent = data.confidence;
        document.getElementById('crop-type-result').textContent = data.crop_type;
        document.getElementById('health-status').textContent = 
            data.is_healthy ? '✅ Healthy' : '⚠️ Disease Detected';

        // Update recommendations
        const recommendations = document.getElementById('recommendations');
        if (data.is_healthy) {
            recommendations.innerHTML = `
                <h3>Maintenance Recommendations</h3>
                <ul>
                    <li>Continue regular monitoring</li>
                    <li>Maintain proper irrigation</li>
                    <li>Ensure adequate nutrients</li>
                    <li>Practice crop rotation</li>
                </ul>
            `;
        } else {
            recommendations.innerHTML = `
                <h3>Treatment Recommendations</h3>
                <ul>
                    <li>Isolate affected plants</li>
                    <li>Consider appropriate treatment</li>
                    <li>Monitor surrounding plants</li>
                    <li>Consult local agricultural expert</li>
                </ul>
            `;
        }
    }
});
