var button = document.querySelector('.button-theme-toggle')
var theme = localStorage.getItem('theme') || 'dark'

function setTheme(type) {
	switch(type) {
		case 'dark':
			document.body.classList.toggle('dark', true); break
		default:
			document.body.classList.toggle('dark', false)
	}
	localStorage.setItem('theme', type)
}

button.addEventListener('click', function(){
	if (theme !== 'dark')
		setTheme((theme='dark'))
	else setTheme((theme='light'))
})

setTheme(theme)