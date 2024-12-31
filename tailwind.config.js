/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html"],
  theme: {
    extend: {
      fontFamily: {
        gallient: ["Gallient", "sans-serif"],
        pangaia: ["Diphylleia", "sans-serif"],
      },
    },
  },
  plugins: [],
};
