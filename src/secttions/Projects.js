import Project from "../components/Project"

const projects = [
	{
		title: "react-chat",
		description: "Simple chat app created with create-react-app deployed with firebase at react-chats-dev.web.app where you can send and recive end to end text with your friend at real time.",
		screenshot: "https://github.com/d33kshant/react-chat/raw/master/screenshot.png",
		source: "https://github.com/d33kshant/react-chat",
		deployment: "https://react-chats-dev.web.app"
	}
]

const Projects = () => {
	return (
		<div className="relative w-full h-full bg-slate-600 flex flex-col p-4">
			<h1 className="text-white text-4xl font-manrope font-bold w-full text-center">Projects</h1>
			<div className="w-full h-full">
				{projects.map(project=><Project {...project} />)}
			</div>
			<button onClick={()=>alert('Left')} className="w-8 h-8 rounded-full absolute top-1/2 left-4 z-10 text-slate-900" >
				<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" fill="currentColor" viewBox="0 0 16 16">
					<path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm3.5 7.5a.5.5 0 0 1 0 1H5.707l2.147 2.146a.5.5 0 0 1-.708.708l-3-3a.5.5 0 0 1 0-.708l3-3a.5.5 0 1 1 .708.708L5.707 7.5H11.5z"/>
				</svg>
			</button>
			<button onClick={()=>alert('Right')} className="w-8 h-8 rounded-full absolute top-1/2 right-4 z-10 text-slate-900" >
				<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" fill="currentColor" viewBox="0 0 16 16">
					<path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/>
				</svg>
			</button>
		</div>
	)
}

export default Projects