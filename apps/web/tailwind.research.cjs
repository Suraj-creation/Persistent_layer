// Training utilities are scoped so the improved assessment workspace keeps its CSS.
module.exports = {
  content: ['./src/components/{TrainingWorkbench,Navbar,ChatInterface,StudentPortal,ResearcherStudio,TrainingResearchPanel,RetrievalExplorer}.tsx'],
  important: '.research-tools',
  corePlugins: { preflight: false },
  theme: { extend: {} },
};
