import Card from './components/Card'
import './styles/App.css'

function App() {
	return (
		<div className="app-container">
			<div className="content-grid">
				<Card className="avatar">
					<img src='https://github.com/d33kshant.png' className='avatar-image' />
				</Card>
				<Card className="intro">
					long intro with description 
				</Card>
				<Card className="social">
					social links
				</Card>
				<Card className="about">
					a brief about
				</Card>
				<Card className="skills">
					all my skills
				</Card>
				<Card className="theme">
					option to change theme					
				</Card>
				<Card className="projects">
					list of my projects
				</Card>
				<Card  className="certificates">
					list of my certificates
				</Card>
			</div>
		</div>
	)
}

export default App
