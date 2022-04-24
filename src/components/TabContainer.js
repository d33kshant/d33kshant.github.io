import "../styles/TabContainer.css"

const TabContainer = ({ children }) => {
	return (
		<div className="main-container">
			<div className="tab-item-container">
				{children}
			</div>
		</div>
	)
}

export default TabContainer