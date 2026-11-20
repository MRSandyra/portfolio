document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-password-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      const field = document.getElementById(button.dataset.passwordToggle);
      if (!field) return;
      const revealing = field.type === 'password';
      field.type = revealing ? 'text' : 'password';
      button.textContent = revealing ? 'Hide' : 'Show';
      button.setAttribute('aria-pressed', String(revealing));
      button.setAttribute('aria-label', `${revealing ? 'Hide' : 'Show'} ${field.labels?.[0]?.textContent || 'password'}`);
    });
  });
});
