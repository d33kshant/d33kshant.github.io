import '../styles/Header.css'

const Header = () => {
	return (
		<header className="main-header">
			<div className="header-container">
				<img className="avatar-image" height={128} width={128} src="https://github.com/d33kshant.png" alt="deekshant's github avatar" />
				<div className="header-info">
					<h1>deekshant</h1>
					<p className="pronouns">He/Him • Student • Full Stack Developer</p>
					student from india doing web development stuffs
					<br />
					with react & javascript related technologies.
					<br />
					<br />
					<div className="header-actions">
						<a href="https://linkedin.com/in/d33kshant">
							<img  height="24" src="https://img.shields.io/badge/Connect-1f6feb?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
						</a>
						<a href="mailto:d33kshant@gmail.com">
							<img height="24" src="https://img.shields.io/badge/Contact-238636?style=flat&logo=gmail&logoColor=white" alt="Twitter Badge"/>
						</a>
					</div>
				</div>
			</div>
		</header>
	)
}

export default Header