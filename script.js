let darkTheme = false

const onThemeChange = () => {

	darkTheme = !darkTheme

	if(darkTheme){
		document.querySelector('body').classList.add('dark')
		document.querySelector('.nav-dark-icon').style.transform = 'translateY(-105%)'
		document.querySelector('.nav-light-icon').style.transform = 'translateY(-50%)'
	}else{
		document.querySelector('body').classList.remove('dark')
		document.querySelector('.nav-dark-icon').style.transform = 'translateY(50%)'
		document.querySelector('.nav-light-icon').style.transform = 'translateY(105%)'
	}
}

document.querySelector('.nav-theme-toggle').addEventListener('click', onThemeChange)