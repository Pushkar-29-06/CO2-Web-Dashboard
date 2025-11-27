// Main JavaScript file for the application

// Initialize the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Application initialized');
    
    // Initialize form validation
    initializeFormValidation();
});

// Initialize form validation
function initializeFormValidation() {
    console.log('Form validation initialized');
    
    // Handle form submission for city comparison
    const compareForm = document.getElementById('compareForm');
    if (compareForm) {
        compareForm.addEventListener('submit', function(e) {
            const city1 = document.getElementById('city1')?.value;
            const city2 = document.getElementById('city2')?.value;
            const errorElement = document.getElementById('error-message') || { textContent: '' };
            
            if (!city1 || !city2) {
                e.preventDefault();
                errorElement.textContent = 'Please select both cities to compare.';
                return false;
            }
            
            if (city1 === city2) {
                e.preventDefault();
                errorElement.textContent = 'Please select two different cities.';
                return false;
            }
            
            errorElement.textContent = '';
            return true;
        });
        console.log('Form event listeners initialized');
    }
}
