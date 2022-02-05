var theme = localStorage.getItem('theme') || 'light'

const setTheme = type => {
	switch(type) {
		case 'dark':
			document.body.classList.add('dark')
			document.body.classList.remove('light')
			break
		default:
			document.body.classList.add('light')
			document.body.classList.remove('dark')
	}
	theme = type
	localStorage.setItem('theme', theme)
}

const toggleTheme = () => {
	if (theme === 'dark') { 
		setTheme('light')
	} else {
		setTheme('dark')
	}
}

document.querySelector('.theme-toggle-button')
.addEventListener('click', toggleTheme)

setTheme(theme)