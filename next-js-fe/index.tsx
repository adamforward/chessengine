import * as React from 'react';
import { useState } from 'react';
import DifficultySelection from './difficulty-selection'; // Ensure this is the correct path
import SideSelection from './side-selection'; // Ensure this is the correct path
import { SelectChangeEvent } from '@mui/material';

const Home: React.FC = () => {
   const [side, setSide] = useState('White');
    const [difficulty, setDifficulty] = useState('Easy');
    const setTeam = useSetRecoilState(teamAtom); 
    setTeam(side);
    const history = history();
  
    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
  
    
    
    history.push('/board');
    };

  return (
    <form onSubmit={handleSubmit}>
        <Box className="App">
      <Box sx= {{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', maxWidth: 400, gap: 5 }}>
        <h1>Let's play chess! Please select difficulty and team.</h1>
        <SideSelection value={side} onChange={(e) => setSide(e.target.value)} />
        <DifficultySelection value={difficulty} onChange={(e) => setDifficulty(e.target.value)} />
        <Button type="submit" variant="contained" color="primary">
          Submit
        </Button>
      </Box>
      </Box>
    </form>
  );
};

export default ChessForm;

export default Home;

function useSetRecoilState(teamAtom: any) {
    throw new Error('Function not implemented.');
}
