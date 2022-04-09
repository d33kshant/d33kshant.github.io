import { motion } from 'framer-motion'

const MainSection = () => {
	return (
		<div className="p-[16px] w-full h-full bg-zinc-200 flex flex-col items-center justify-center relative">
			<motion.h3
				initial={{ y: -16, opacity: 0 }}
				animate={{ y: 0, opacity: 1 }}
				transition={{ duration: 0.5 }}
				className="text-5xl md:text-7xl font-bold font-manrope text-zinc-900"
			>
				deekshant
			</motion.h3>
			<motion.p
				initial={{ y: -16, opacity: 0 }}
				animate={{ y: 4, opacity: 1 }}
				transition={{ delay: 0.5, duration: 0.5 }}
				className="font-manrope max-w-2xl text-center text-lg mt-1 md:m-0 md:text-xl text-zinc-500"
			>
				He/Him studying computer science and trying to make cool thing on the web for the web with react and other stuffs.
			</motion.p>
			<motion.div
				initial={{ y: 0, opacity: 0 }}
				animate={{ y: 32., opacity: 1 }}
				transition={{ delay: 1, duration: 0.5 }}
				className="flex gap-4 items-center"
			>
				<a href="https://linkedin.com/in/d33kshant" className="font-monospace px-4 py-0.5 text-lg bg-emerald-500">Connect</a>
				<a href="mailto:d33kshant@gmail.com" className="font-monospace px-4 py-0.5 text-lg bg-blue-500">Contact</a>
			</motion.div>
		</div>
	)
}

export default MainSection