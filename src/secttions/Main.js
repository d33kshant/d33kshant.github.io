import { motion } from 'framer-motion'

const MainSection = () => {
	return (
		<div className="p-[16px] w-full h-full bg-[#f1faee] text-[#1d3557] flex flex-col items-center justify-center relative">
			<motion.h3
				initial={{ y: -16, opacity: 0 }}
				animate={{ y: 0, opacity: 1 }}
				transition={{ duration: 0.5 }}
				className="text-[64px] font-bold font-manrope"
			>
				Deekshant
			</motion.h3>
			<motion.p
				initial={{ y: -16, opacity: 0 }}
				animate={{ y: 0, opacity: 1 }}
				transition={{ delay: 0.5, duration: 0.5 }}
				className="font-sans font-semibold text-[#e63946]"
			>
				He/Him • Student • Full Stack Web Developer
			</motion.p>
		</div>
	)
}

export default MainSection