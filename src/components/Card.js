import "../styles/Card.css"

const Card = ({ className, children, ...props }) => {
	return (
		<div className={`card-container ${className}`} {...props} >
			{children}
		</div>
	)
}

export default Card