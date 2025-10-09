// Contributor Models
export interface Contributor {
  id: string;
  name: string;
  title: string;
  bio: string;
  roles: string[];
  avatarUrl?: string;
  linkedInUrl?: string;
  resumeUrl?: string;
}

// Hardcoded contributor data
export const contributors: Contributor[] = [
  {
    id: '1',
    name: 'Harry Nguyen',
    title: 'Full Stack AI Engineer',
    bio: 'Engineer capable of building entire AI-powered products end-to-end with specialization in agentic AI and LLM systems.',
    roles: ['Frontend Development', 'Backend Development', 'AI Engineering', 'CI/CD & Infrastructure'],
    avatarUrl: undefined, // Will use initials fallback
    linkedInUrl: 'https://www.linkedin.com/in/harry-nguyen-wayweaver/'
  },
  {
    id: '2',
    name: 'Amelia Nguyen',
    title: 'Technical Program Manager',
    bio: 'Manager who excels at delivering Fintech and Agentic AI solutions that bridge technology, finance, and operations.',
    roles: ['Scope & Planning', 'Architecture & Design', 'Implementation Standards', 'Integration Management'],
    avatarUrl: undefined,
    linkedInUrl: 'https://www.linkedin.com/in/amelianguyendo/',
    resumeUrl: 'https://standardresume.co/r/aphr58wioGS3ds5b_FU8b'
  }
];
