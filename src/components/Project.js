const Project = ({ title, description, screenshot, source, deployment }) => {
	return (
		<div className="w-full flex flex-col justify-center items-center gap-4">
			<h1 className="font-manrope font-bold text-2xl" >{title}</h1>
			<p className="font-sans text-lg max-w-3xl text-center">{description}</p>
			<img src={screenshot} alt={title} width={540} />
			<div className="flex gap-2" >
				<a href={source}>
					Source Code
				</a>
				<a href={deployment}>
					Open App
				</a>
			</div>
		</div>
	)
}

export default Project