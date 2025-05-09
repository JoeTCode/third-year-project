function previewImage(event) {
      const reader = new FileReader();
      reader.onload = () => {
        const output = document.getElementById('image-preview');
        output.src = reader.result;
        output.style.display = 'block';
      };
      reader.readAsDataURL(event.target.files[0]);
}