import data from '../data.json'
import "../styles/Projects.css"
import Project from './Project'

const Projects = () => {
	return (
		<>
		<div className="projects-grid">
			{data.projects.map((project, index)=><Project key={index} {...project} />)}
		</div>
		<hr className='divider' />
		<span className='project-section-footer'>Most of my projects are available on <a href='https://github.com/d33kshant?tab=repositories'>GitHub</a></span>
		</>
	)
}

export default Projects