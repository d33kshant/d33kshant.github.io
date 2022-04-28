import "../styles/Project.css"

const Project = ({title, description, source, icon}) => {
	return (
		<a href={source} className="project-container">
			<div className="content">
				<div className="project-thumb">
					<img src={icon} alt={title} />
				</div>
				<h3>{title}</h3>
				<p>{description}</p>
			</div>
		</a>
	)
}

export default Project