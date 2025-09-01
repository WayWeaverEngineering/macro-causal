import './app/style/styles.css'
import { createRoot } from 'react-dom/client'
import { Provider } from 'react-redux'
import MacroAnalyst from './app/layout/MacroAnalyst';
import { store } from './redux/store';

createRoot(document.getElementById('root')!).render(
  <Provider store={store}>
    <MacroAnalyst />
  </Provider>
)