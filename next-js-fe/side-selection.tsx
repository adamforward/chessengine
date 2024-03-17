import * as React from 'react';
import Box from '@mui/material/Box';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select, { SelectChangeEvent } from '@mui/material/Select';

interface SideSelectionProps {
  value: string;
  onChange: (event: SelectChangeEvent<string>) => void;
}

const SideSelection: React.FC<SideSelectionProps> = ({ value, onChange }) => {
  return (
    <Box sx={{ minWidth: 400 }}>
      <FormControl fullWidth>
        <InputLabel id="team-select-label">Team</InputLabel>
        <Select
          labelId="team-select-label"
          id="team-select"
          value={value}
          label="Team"
          onChange={onChange}
        >
          <MenuItem value="White">White</MenuItem>
          <MenuItem value="Black">Black</MenuItem>
        </Select>
      </FormControl>
    </Box>
  );
};

export default SideSelection;