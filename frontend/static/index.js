function previewImage(event) {
      const reader = new FileReader();
      reader.onload = () => {
        const output = document.getElementById('image-preview');
        output.src = reader.result;
        output.style.display = 'block';
      };
      reader.readAsDataURL(event.target.files[0]);
}

function toggleRotateInput() {
    const checkbox = document.getElementById('rotate');
    const input = document.getElementById('rotate-angle');
    input.style.display = checkbox.checked ? 'inline' : 'none';
}