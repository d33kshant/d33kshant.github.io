const Project = ({ title, description, screenshot, source, deployment }) => {
	return (
		<div className="w-full flex flex-col justify-center items-center gap-4">
			<h1 className="font-manrope font-bold text-2xl" >{title}</h1>
			<p className="font-sans text-lg max-w-3xl text-center">{description}</p>
			<div className="flex gap-2" >
				<a href={source} className="bg-slate-400 text-slate-600 rounded px-4 py-2">
					Source Code
				</a>
				<a href={deployment} className="bg-slate-400 text-slate-600 rounded px-4 py-2">
					Open App
				</a>
			</div>
		</div>
	)
}

export default Project