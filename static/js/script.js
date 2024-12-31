function displayVideo(event) {
  const file = event.target.files[0];
  const placeholderText = document.getElementById("placeholder-text");
  const videoPreview = document.getElementById("video-preview");
  const closeButton = document.getElementById("close-button");

  if (file) {
    const videoURL = URL.createObjectURL(file);

    // Hide the placeholder text
    placeholderText.style.display = "none";

    // Show the video element
    videoPreview.src = videoURL;
    videoPreview.classList.remove("hidden");
    closeButton.classList.remove("hidden");
    closeButton.classList.add("flex");
  }
}

function submitVideo(event) {
  event.preventDefault(); // Prevent default form submission

  const form = document.getElementById("video-form");
  const formData = new FormData(form);
  const outputText = document.getElementById("output-text");
  const loadingSpinner = document.getElementById("loading-spinner");
  const buttons = document.querySelectorAll("button, label");

  // Show the loader and disable buttons
  loadingSpinner.classList.remove("hidden");
  buttons.forEach((button) => (button.disabled = true));

  fetch("http://127.0.0.1:5000/upload-video", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to process the video");
      }
      return response.json(); // Ensure JSON response
    })
    .then((data) => {
      outputText.textContent = data.message || "Translation received!";
    })
    .catch((error) => {
      console.error("Error:", error);
      outputText.textContent = "An error occurred. Please try again.";
    })
    .finally(() => {
      // Hide the loader and re-enable buttons
      loadingSpinner.classList.add("hidden");
      buttons.forEach((button) => (button.disabled = false));
    });
}

// Attach the submitVideo function to the form
document.getElementById("video-form").addEventListener("submit", submitVideo);
