let darkMode = JSON.parse(localStorage.getItem('dark-mode')) || false

const setTheme = dark => {
	if (!dark) {
		document.documentElement.style.setProperty('color-scheme', 'dark')
		document.documentElement.classList.add('dark')
	} else {
		document.documentElement.style.setProperty('color-scheme', 'light')
		document.documentElement.classList.remove('dark')
	}
	localStorage.setItem('dark-mode', dark)
	return !dark
}

let theme = setTheme(darkMode)

document.querySelector('.toggle-theme').addEventListener('click', (event)=>{
	theme = setTheme(theme)
})