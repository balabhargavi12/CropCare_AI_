// Auth page functionality
document.addEventListener('DOMContentLoaded', function() {
    // Setup password validation listeners if we're on the register page
    const registerPassword = document.getElementById('register-password');
    const confirmPassword = document.getElementById('register-confirm-password');
    const passwordStrengthIndicator = document.getElementById('password-strength');
    const passwordMatchIndicator = document.getElementById('password-match');
    const registerBtn = document.getElementById('register-btn');
    
    if (registerPassword && confirmPassword) {
        registerPassword.addEventListener('input', validatePassword);
        confirmPassword.addEventListener('input', validatePasswordMatch);
    }
    
    // Check URL hash for direct registration tab
    if (window.location.hash === '#register') {
        showTab('register');
    }
});

function showTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');

    // Update form containers
    document.querySelectorAll('.form-container').forEach(form => {
        form.classList.remove('active');
    });
    document.getElementById(`${tabName}-form`).classList.add('active');
    
    // Update URL hash
    window.location.hash = tabName;
}

function validatePassword() {
    const password = document.getElementById('register-password').value;
    const strengthIndicator = document.getElementById('password-strength');
    const registerBtn = document.getElementById('register-btn');
    
    // Check password strength
    let strength = 0;
    if (password.length >= 8) strength += 1;
    if (password.match(/[A-Z]/)) strength += 1;
    if (password.match(/[0-9]/)) strength += 1;
    if (password.match(/[^A-Za-z0-9]/)) strength += 1;
    
    // Display strength message
    switch(strength) {
        case 0:
            strengthIndicator.textContent = 'Password is too weak';
            strengthIndicator.className = 'password-strength weak';
            break;
        case 1:
        case 2:
            strengthIndicator.textContent = 'Password is medium strength';
            strengthIndicator.className = 'password-strength medium';
            break;
        case 3:
        case 4:
            strengthIndicator.textContent = 'Password is strong';
            strengthIndicator.className = 'password-strength strong';
            break;
    }
    
    // Check both conditions before enabling register button
    validateFormCompletion();
}

function validatePasswordMatch() {
    const password = document.getElementById('register-password').value;
    const confirmPassword = document.getElementById('register-confirm-password').value;
    const matchIndicator = document.getElementById('password-match');
    
    if (!confirmPassword) {
        matchIndicator.textContent = '';
        return;
    }
    
    if (password === confirmPassword) {
        matchIndicator.textContent = 'Passwords match';
        matchIndicator.className = 'password-match match';
    } else {
        matchIndicator.textContent = 'Passwords do not match';
        matchIndicator.className = 'password-match no-match';
    }
    
    // Check both conditions before enabling register button
    validateFormCompletion();
}

function validateFormCompletion() {
    const username = document.getElementById('register-username').value;
    const password = document.getElementById('register-password').value;
    const confirmPassword = document.getElementById('register-confirm-password').value;
    const registerBtn = document.getElementById('register-btn');
    
    // Enable button only if all fields are filled and passwords match
    if (username && password && confirmPassword && password === confirmPassword && password.length >= 8) {
        registerBtn.disabled = false;
    } else {
        registerBtn.disabled = true;
    }
}

async function handleLogin(event) {
    event.preventDefault();

    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();

        if (data.success) {
            window.location.href = '/dashboard';
        } else {
            showError('login-form', data.message || 'Login failed. Please check your credentials.');
        }
    } catch (error) {
        console.error('Login error:', error);
        showError('login-form', 'An error occurred during login. Please try again.');
    }
}

async function handleRegister(event) {
    event.preventDefault();

    const username = document.getElementById('register-username').value;
    const password = document.getElementById('register-password').value;
    const confirmPassword = document.getElementById('register-confirm-password').value;
    
    // Double check passwords match
    if (password !== confirmPassword) {
        showError('register-form', 'Passwords do not match.');
        return;
    }

    try {
        const response = await fetch('/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();

        if (data.success) {
            showSuccess('Registration successful! Please login with your new credentials.');
            setTimeout(() => {
                showTab('login');
            }, 1500);
        } else {
            showError('register-form', data.message || 'Registration failed');
        }
    } catch (error) {
        console.error('Registration error:', error);
        showError('register-form', 'An error occurred during registration');
    }
}

function showError(formId, message) {
    // Remove any existing error messages
    const existingError = document.querySelector('.error-message');
    if (existingError) {
        existingError.remove();
    }
    
    // Create and show new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    const form = document.getElementById(formId);
    form.querySelector('button').insertAdjacentElement('beforebegin', errorDiv);
}

function showSuccess(message) {
    // Remove any existing messages
    const existingMessage = document.querySelector('.success-message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    // Create and show success message
    const successDiv = document.createElement('div');
    successDiv.className = 'success-message';
    successDiv.textContent = message;
    
    document.querySelector('.auth-container').prepend(successDiv);
}
