import * as React from 'react';
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select, { SelectChangeEvent } from '@mui/material/Select';

interface DifficultySelectionProps {
  value: string;
  onChange: (event: SelectChangeEvent<string>) => void;
}

const DifficultySelection: React.FC<DifficultySelectionProps> = ({ value, onChange }) => {
  return (
    <Box sx={{ minWidth: 400 }}>
      <FormControl fullWidth>
        <InputLabel id="difficulty-select-label">Difficulty</InputLabel>
        <Select
          labelId="difficulty-select-label"
          id="difficulty-select"
          value={value}
          label="Difficulty"
          onChange={onChange}
        >
          <MenuItem value="Easiest">Easiest</MenuItem>
          <MenuItem value="Easy">Easy</MenuItem>
          <MenuItem value="Medium">Medium</MenuItem>
          <MenuItem value="Hard">Hard</MenuItem>
          <MenuItem value="Hardest">Hardest</MenuItem>
        </Select>
      </FormControl>
    </Box>
  );
};

export default DifficultySelection;
