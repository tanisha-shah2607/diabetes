// Initialize Feather icons
feather.replace();

// Form submission handling
document.getElementById('predictionForm').addEventListener('submit', function(e) {
    const submitBtn = document.getElementById('submitBtn');
    const spinner = submitBtn.querySelector('.spinner-border');
    
    // Show loading state
    spinner.classList.remove('d-none');
    submitBtn.disabled = true;
    
    // Form will submit normally
});

// Initialize tooltips
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
});
