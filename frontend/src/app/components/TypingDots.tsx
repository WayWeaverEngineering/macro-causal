import { Box } from "@mui/material";

interface TypingDotsProps {
  color?: string;
}

export const TypingDots = ({ color = "#90caf9" }: TypingDotsProps) => {
  return (
    <Box
      sx={{
        display: "flex",
        gap: "4px",
        alignItems: "center",
        height: "1em",
        justifyContent: "flex-start"
      }}
    >
      {[0, 1, 2].map((i) => (
        <Box
          key={i}
          sx={{
            width: "6px",
            height: "6px",
            borderRadius: "50%",
            backgroundColor: color,
            opacity: 0,
            animation: `typingDots 1.4s infinite`,
            animationDelay: `${i * 0.2}s`
          }}
        />
      ))}

      <style>
        {`
          @keyframes typingDots {
            0% { opacity: 0; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1); }
            100% { opacity: 0; transform: scale(0.8); }
          }
        `}
      </style>
    </Box>
  );
};
