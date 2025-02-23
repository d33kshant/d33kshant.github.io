import better

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'd33kshant'
copyright = '2025, Deekshant Yadav'
author = 'Deekshant Yadav'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser']
source_suffix = {
  '.rst': 'restructuredtext',
  '.md': 'markdown',
}
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'env']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'better'
html_theme_path = [better.better_theme_path]
html_static_path = ['_static']
html_short_title = "Home"
html_title = project
html_show_sphinx = False
html_show_copyright = True

html_theme_options = {
  # show sidebar on the right instead of on the left
  'rightsidebar': False,

  # inline CSS to insert into the page if you're too lazy to make a
  # separate file
  'inlinecss': '',

  # CSS files to include after all other CSS files
  # (refer to by relative path from conf.py directory, or link to a
  # remote file)
  'cssfiles': ['_static/styles.css'],

  # show a big text header with the value of html_title
  'showheader': True,

  # show the breadcrumbs and index|next|previous links at the top of
  # the page
  'showrelbartop': True,
  # same for bottom of the page
  'showrelbarbottom': True,

  # show the self-serving link in the footer
  'linktotheme': False,

  # width of the sidebar. page width is determined by a CSS rule.
  # I prefer to define things in rem because it scales with the
  # global font size rather than pixels or the local font size.
  'sidebarwidth': '15rem',

  # color of all body text
  'textcolor': 'black',

  # color of all headings (<h1> tags); defaults to the value of
  # textcolor, which is why it's defined here at all.
  'headtextcolor': 'black',

  # color of text in the footer, including links; defaults to the
  # value of textcolor
  'footertextcolor': 'black',

  # Google Analytics info
  'ga_ua': '',
  'ga_domain': '',
}