let darkTheme = localStorage.getItem('dark-theme') === 'true' || false

const onThemeChange = () => {	
	
	if(darkTheme){
		document.querySelector('body').classList.add('dark')
		document.querySelector('.nav-dark-icon').style.transform = 'translateY(-105%)'
		document.querySelector('.nav-light-icon').style.transform = 'translateY(-50%)'
	}else{
		document.querySelector('body').classList.remove('dark')
		document.querySelector('.nav-dark-icon').style.transform = 'translateY(50%)'
		document.querySelector('.nav-light-icon').style.transform = 'translateY(105%)'
	}
	
	localStorage.setItem('dark-theme', darkTheme.toString())
}

const toggleTheme = () => {
	darkTheme = !darkTheme
	onThemeChange()
}

document.querySelector('.nav-theme-toggle').addEventListener('click', toggleTheme)
onThemeChange()