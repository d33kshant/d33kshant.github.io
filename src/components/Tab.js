import "../styles/Tab.css"

const Tab = ({ active, color, children, onClick }) => {
	return (
		<div onClick={onClick} className={`tab ${active && 'active-tab'}`} >
			{children}
		</div>
	)
}

export default Tab