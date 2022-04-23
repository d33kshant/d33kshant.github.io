import '../styles/TabBar.css'

const TabBar = ({ children }) => {
	return (
		<nav className="main-tab">
			<div className="tab-container">
				{children}
			</div>
		</nav>
	)
}

export default TabBar