// Add interactivity to the website

// Contact form submission handling
document.getElementById('contact-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the form from reloading the page

    const name = event.target.name.value;
    const email = event.target.email.value;
    const message = event.target.message.value;

    alert(`Thanks for reaching out, ${name}! We will respond to your message soon.`);

    // Clear form fields
    event.target.reset();
});

// Smooth scrolling for navigation links
document.querySelectorAll('nav ul li a').forEach(anchor => {
    anchor.addEventListener('click', function(event) {
        event.preventDefault();

        const targetId = this.getAttribute('href').substring(1);
        document.getElementById(targetId).scrollIntoView({
            behavior: 'smooth'
        });
    });
});