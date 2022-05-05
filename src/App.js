import { useState, useEffect } from 'react'
import Certificates from './components/Certificates'
import Header from './components/Header'
import Projects from './components/Projects'
import Skills from './components/Skils'
import Tab from './components/Tab'
import TabBar from './components/TabBar'
import TabContainer from './components/TabContainer'
import './styles/App.css'

const tabs = [ 
	{
		title: "Projects",
		icon: <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
	},
	{
		title: "Skills",
		icon: <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
	},
	{
		title: "Certificates",
		icon: <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round"><rect x="1" y="4" width="22" height="16" rx="2" ry="2"></rect><line x1="1" y1="10" x2="23" y2="10"></line></svg>
	}
]

const tabComponents = [ <Projects/>, <Skills/>, <Certificates/> ]

function App() {

	const [currentTab, setCurrentTab] = useState(0)
	const [theme, setTheme] = useState('')

	useEffect(() => {
		const _theme = localStorage.getItem('theme')
		if (_theme) setTheme(_theme)
		document.body.classList.toggle('dark', _theme!=='dark')
	}, [])

	const toggleTheme = () => {
		if (theme === 'light') {
			setTheme('dark')
		} else setTheme('light')
		localStorage.setItem('theme', theme)
		document.body.classList.toggle('dark', theme!=='dark')
	}

	return (
		<>
		<Header />
		<TabBar>
			{ tabs.map((tab, index)=><Tab active={currentTab===index} onClick={()=>setCurrentTab(index)} key={index}>{tab.icon}{tab.title}</Tab>) }
			<button onClick={toggleTheme} className="enable-dark-mode">
				<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" strokeWidth="2" fill="currentColor" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
			</button>
		</TabBar>
		<TabContainer>
			{tabComponents[currentTab]}
		</TabContainer>
		</>
	)
}

export default App