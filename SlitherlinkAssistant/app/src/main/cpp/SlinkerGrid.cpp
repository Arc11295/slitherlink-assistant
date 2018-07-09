/*
	Slinker - Copyright (C) 2008 Tim Hutton, tim.hutton@gmail.com, http://www.sq3.org.uk

	This file is part of Slinker.

	Slinker is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Slinker is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with Slinker.  If not, see <http://www.gnu.org/licenses/>.
*/

// local:
#include "SlinkerGrid.h"

// STL:
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <cmath>

using namespace std;

/// ---------- statics -------------

const int SlinkerGrid::UNKNOWN = -1;
// (we should be able to put this value in the header file but gcc doesn't like it)

const int SlinkerGrid::NHOOD[4][2] = { {-1,0}, { 0,-1}, {1,0}, {0,1} };

const int SlinkerGrid::N_SYMMETRIES = 8;

const SlinkerGrid::TMatrix SlinkerGrid::SYMMETRIES[N_SYMMETRIES] = {
	TMatrix(1,0,0,1), // identity
	TMatrix(0,-1,1,0), // rotation by 90 degrees ccwise
	TMatrix(-1,0,0,-1), // rotation by 180 degrees
	TMatrix(0,1,-1,0), // rotation by 270 degrees ccwise
	TMatrix(-1,0,0,1), // reflection in x
	TMatrix(0,1,1,0), // reflection in x then 90 degrees ccwise
	TMatrix(1,0,0,-1), // reflection in x then 180 degrees
	TMatrix(0,-1,-1,0) // reflection in x then 270 degrees ccwise
};

bool SlinkerGrid::IsOdd(int a) { return (abs(a)&1)==1; } // abs here to extend test to negative numbers safely
bool SlinkerGrid::IsEven(int a) { return (abs(a)&1)==0; } 
bool SlinkerGrid::IsDot(int x,int y) { return( IsEven(x) && IsEven(y) ); }
bool SlinkerGrid::IsBorder(int x,int y) { return( IsEven(x)^IsEven(y) ); }
bool SlinkerGrid::IsCell(int x,int y) { return( IsOdd(x) && IsOdd(y) ); }
bool SlinkerGrid::IsHorizontalBorder(int x,int y) { return( IsOdd(x) && IsEven(y) ); }
bool SlinkerGrid::IsVerticalBorder(int x,int y) { return( IsEven(x) && IsOdd(y) ); }

// ---------------------------

SlinkerGrid::SlinkerGrid() : X(0),Y(0),grid_shape(RectangleShape)
{
}

SlinkerGrid::SlinkerGrid(int x,int y,TGridShape gs) 
	: X(x), Y(y), 
	grid_shape(gs)
{
	cells.assign(2*X+1,vector<int>(2*Y+1,UNKNOWN)); // init array of values all filled with UNKNOWN
}

SlinkerGrid::SlinkerGrid(const SlinkerGrid& g)
	: X(g.X),Y(g.Y),cells(g.cells),grid_shape(g.grid_shape)
{
}

SlinkerGrid& SlinkerGrid::operator=(const SlinkerGrid& g)
{
	X = g.X;
	Y = g.Y;
	cells = g.cells;
	grid_shape = g.grid_shape;
	return *this;
}

bool SlinkerGrid::operator==(const SlinkerGrid& g) const
{
	return(X==g.X && Y==g.Y && grid_shape==g.grid_shape && cells==g.cells);
}

int myMax(int a,int b) { return (a>b)?a:b; }

bool SlinkerGrid::IsOnGrid(int x,int y) const
{
	// even non-rectangular grids are embedded in a rectangular area
	if(!(x>=0 && x<2*X+1 && y>=0 && y<2*Y+1))
		return false;

	switch(this->grid_shape)
	{
		case RectangleShape:
			{
				return true; // plain rectangle
			}
		case MissingCentre:
			{
				// rectangle with missing central ninth
				// (looks best if X and Y are divisible by 3)
				return (x<2*int(X/3)+1 || x>2*(2*int(X/3)-1)+1 || y<2*int(Y/3)+1 || y>2*(2*int(Y/3)-1)+1);
			}
		case CircleShape:
			{
				// TODO: make a test x and y, based on a square (an oval is a flattened circle)
				int tx=x,ty=y;
				//if(X<Y) tx = tx*Y/X; // TODO: this isn't quite right, we want symmetrical ovals
				//else if(Y<X) ty = ty*X/Y;
				int mid = (2*myMax(X,Y)+1)/2;
				int r = mid; // radius of inset circle
				
				if(false)
				{ // DEBUG
					ostringstream oss;
					oss << X << "," << Y << " grid:\n\n" << x << "," << y << " becomes " << tx << "," 
						<< ty << "\n\nmid=r=" << mid;
				}
				if(IsCell(x,y))
					return(hypot(mid-tx,mid-ty)<=r);
				else if(IsHorizontalBorder(x,y))
					return(hypot(mid-tx,mid-(ty-1))<=r || hypot(mid-tx,mid-(ty+1))<=r);
				else if(IsVerticalBorder(x,y))
					return(hypot(mid-(tx-1),mid-ty)<=r || hypot(mid-(tx+1),mid-ty)<=r);
				else if(IsDot(x,y))
					return(hypot(mid-(tx-1),mid-(ty-1))<=r || hypot(mid-(tx+1),mid-(ty-1))<=r ||
						hypot(mid-(tx+1),mid-(ty+1))<=r || hypot(mid-(tx-1),mid-(ty+1))<=r);
			}
	}
	throw(runtime_error("Internal error: unknown grid type in IsOnGrid."));
	return false;
}

bool SlinkerGrid::IsLegal() const
{
	// test 1: does the border count match the cell value for every cell?
	int x,y;
	int cell_val,min_borders,max_borders;
	for(x=0;x<2*X+1;x++)
	{
		for(y=0;y<2*Y+1;y++)
		{
			if(!IsOnGrid(x,y) || !IsCell(x,y)) continue;
			cell_val = cells[x][y];
			// is cell value is unknown then this square is legal by this test
			if(cell_val == UNKNOWN) continue;
			// retrieve the possible range (given unknowns) of borders this cell might have
			GetBorderCountAroundCell(x,y,min_borders,max_borders);
			if(cell_val<min_borders || cell_val>max_borders)
				return false;
		}
	}
	// test 2: does any dot have the wrong number of borders joining it?
	for(x=0;x<2*X+1;x+=2)
	{
		for(y=0;y<2*Y+1;y+=2)
		{
			if(!IsOnGrid(x,y)) continue;
			GetBorderCountAroundDot(x,y,min_borders,max_borders);
			if(min_borders>2) // too many borders joining!
				return false;
			if(min_borders>0 && max_borders<2) // only one border comes here!
				return false;
		}
	}
	// test 3: all borders are connected
	{
		// count how many borders there are, and pick a dot that has a border
		int n_borders=0;
		vector< pair<int,int> > dots;
		for(x=0;x<2*X+1;x++)
		{
			for(y=0;y<2*Y+1;y++)
			{
				if(!IsOnGrid(x,y)) continue;
				if(IsBorder(x,y) && cells[x][y]==1) 
				{
					n_borders++;
					if(dots.empty())
					{
						if(IsHorizontalBorder(x,y))
							dots.push_back(make_pair(x-1,y));
						else
							dots.push_back(make_pair(x,y-1));
					}
				}
			}
		}
		if(n_borders==0) 
			return false; // a grid with no borders could pass tests 1 and 2 but we disallow it here
		CollectJoinedDots(dots);
		if(dots.size()!=n_borders)
			return false; // found a small loop
	}

	return true; // didn't find anything wrong
}

void SlinkerGrid::CollectJoinedDots(vector< pair<int,int> > &dots) const
{
	pair<int,int> current_dot,next_dot;
	int dir;
	int n_added;
	do
	{
		current_dot = dots.back();
		n_added=0;
		for(dir=0;dir<4;dir++)
		{
			next_dot = make_pair(current_dot.first + NHOOD[dir][0]*2,current_dot.second + NHOOD[dir][1]*2);
			if(IsOnGrid(next_dot.first,next_dot.second) && 
				cells[current_dot.first + NHOOD[dir][0]][current_dot.second + NHOOD[dir][1]]==1 &&
				find(dots.begin(),dots.end(),next_dot)==dots.end())
			{
				dots.push_back(next_dot);
				n_added++;
			}
		}
	} while(n_added>0);
}

void SlinkerGrid::MarkUnknownBordersAsOff()
{
	int x,y;
	for(x=0;x<2*X+1;x++)
	{
		for(y=0;y<2*Y+1;y++)
		{
			if(IsBorder(x,y) && cells[x][y]==UNKNOWN)
				cells[x][y] = 0;
		}
	}
}

void SlinkerGrid::MarkOffBordersAsUnknown()
{
	int x,y;
	for(x=0;x<2*X+1;x++)
	{
		for(y=0;y<2*Y+1;y++)
		{
			if(IsBorder(x,y) && cells[x][y]==0)
				cells[x][y] = UNKNOWN;
		}
	}
}

bool SlinkerGrid::IsFinished() const
{
	// not finished if any border is marked as UNKNOWN
	int x,y;
	for(x=0;x<2*X+1;x++)
	{
		for(y=0;y<2*Y+1;y++)
		{
			if(IsOnGrid(x,y) && IsBorder(x,y) && cells[x][y]==UNKNOWN)
				return false;
		}
	}
	return true;
} 

int& SlinkerGrid::cellValue(int x,int y)
{
	if(x<0 || x>=X || y<0 || y>=Y)
	{
		ostringstream oss;
		oss << "out of range exception in SlinkerGrid::cellValue : " << x << "," << y;
		throw(out_of_range(oss.str().c_str()));
	}
	return this->cells[2*x+1][2*y+1];
}

const int& SlinkerGrid::cellValue(int x,int y) const
{
	if(x<0 || x>=X || y<0 || y>=Y)
	{
		ostringstream oss;
		oss << "out of range exception in SlinkerGrid::cellValue : " << x << "," << y;
		throw(out_of_range(oss.str().c_str()));
	}
	return this->cells[2*x+1][2*y+1];
}

int& SlinkerGrid::gridValue(int x,int y)
{
	if(!IsOnGrid(x,y))
	{
		ostringstream oss;
		oss << "out of range exception in SlinkerGrid::gridValue : " << x << "," << y;
		throw(out_of_range(oss.str().c_str()));
	}
	return this->cells[x][y];
}

const int& SlinkerGrid::gridValue(int x,int y) const
{
	if(!IsOnGrid(x,y))
		throw(out_of_range("out of range exception in SlinkerGrid::gridValue"));
	return this->cells[x][y];
}

void SlinkerGrid::GetBorderCountAroundCell(int x,int y,int &min,int &max) const
{
	if(x<1 || x>=2*X || y<1 || y>=2*Y)
		throw(out_of_range("out of range exception in SlinkerGrid::GetBorderCountAroundCell"));
	if(!IsCell(x,y))
		throw(out_of_range("non-cell coords in SlinkerGrid::GetBorderCountAroundCell"));
	min = 0;
	max = 4;
	int b;
	for(int dir=0;dir<4;dir++)
	{
		b = cells[x+NHOOD[dir][0]][y+NHOOD[dir][1]];
		if(b==1)
			min++;
		if(b==0)
			max--;
	}
}

void SlinkerGrid::GetBorderCountAroundDot(int x,int y,int &min,int &max) const
{
	if(!IsOnGrid(x,y))
		throw(out_of_range("out of range exception in SlinkerGrid::GetBorderCountAroundDot"));
	if(!IsDot(x,y))
		throw(out_of_range("non-dot coordinates in SlinkerGrid::GetBorderCountAroundDot"));
	min = 0;
	max = 4;
	int tx,ty,b;
	for(int dir=0;dir<4;dir++)
	{
		tx = x+NHOOD[dir][0];
		ty = y+NHOOD[dir][1];
		// this location might be off the grid
		if(!IsOnGrid(tx,ty))
			max--;
		else {
			b = cells[tx][ty];
			if(b==1) 
				min++;
			else if(b==0)
				max--;
		}
	}
}

string SlinkerGrid::GetPrintOut() const
{
	ostringstream oss;
	int x,y,v;
	// make a text representation (looks best rendered in a fixed-width font)
	for(y=0;y<2*Y+1;y++)
	{
		for(x=0;x<2*X+1;x++)
		{
			if(!IsOnGrid(x,y)) // allowing for non-rectangular grids
			{
				oss << "#";
			}
			else
			{
				v = cells[x][y];
				if(IsDot(x,y))
					oss << "+";
				else if(v==UNKNOWN) // unknowns we leave blank
					oss << " ";
				else if(IsHorizontalBorder(x,y))
				{
					if(v)
						oss << "-";
					else
						oss << "x";
				}
				else if(IsVerticalBorder(x,y))
				{
					if(v)
						oss << "|";
					else
						oss << "x";
				}
				else // is a known cell entry
					oss << v;
			}
			oss << "  ";
		}
		oss << "\n";
	}
	return oss.str();
}

void SlinkerGrid::Clear()
{
	int x;
	for(x=0;x<2*X+1;x++)
	{
		fill(cells[x].begin(),cells[x].end(),UNKNOWN); 
	}
}

void SlinkerGrid::ClearBorders()
{
	int x,y;
	for(x=0;x<2*X+1;x++)
	{
		for(y=0;y<2*Y+1;y++)
		{
			if(IsOnGrid(x,y) && IsBorder(x,y))
				this->cells[x][y] = UNKNOWN;
		}
	}
}

bool SlinkerGrid::ApplyRules(const vector<TRule>& rules,vector<int*> &changed)
{
	int x,y;
	unsigned int iSymm;
	bool can_apply,did_something;
	vector<TElement>::const_iterator it;
	vector<TRule>::const_iterator rule_it;
	int tx,ty;
	changed.clear(); // don't want any previous entries
	int *pEntry;
	do 
	{
		did_something = false;
		for(rule_it = rules.begin();rule_it!=rules.end();rule_it++)
		{
			for(x=-1;x<2*X+1+1;x++) // consider rule applications to the cells around the edge too (see the elementary rules)
			{
				for(y=-1;y<2*Y+1+1;y++)
				{
					if(!IsCell(x,y)) continue;
					// does the rule apply here (centred on cell x,y) in any orientation?
					for(iSymm=0;iSymm<N_SYMMETRIES;iSymm++)
					{
						can_apply = true;
						for(it=rule_it->required.begin();it!=rule_it->required.end() && can_apply;it++)
						{
							tx = x + SYMMETRIES[iSymm].mX(it->x,it->y);
							ty = y + SYMMETRIES[iSymm].mY(it->x,it->y);
							if( !( (it->val==0 && IsBorder(tx,ty) && !IsOnGrid(tx,ty)) || 
								(IsOnGrid(tx,ty) && cells[tx][ty]==it->val) ) )
									can_apply = false;
						}
						if(!can_apply) continue;
						for(it = rule_it->implied.begin();it!=rule_it->implied.end();it++)
						{
							tx = x + SYMMETRIES[iSymm].mX(it->x,it->y);
							ty = y + SYMMETRIES[iSymm].mY(it->x,it->y);
							if(!IsOnGrid(tx,ty)) continue;
							pEntry = &cells[tx][ty];
							if(*pEntry != it->val)
							{
								if(*pEntry != UNKNOWN)
								{
									// We've found a contradiction! this means the grid was inconsistent.
									// In FollowPossibilities this means we can abandon this search avenue.
									// In FindNewRules this means we may have found a rule.
									return false; 
								}
								*pEntry = it->val;
								changed.push_back(pEntry);
								did_something = true;
							}
						}
					}
				}
			}
		}
	} while(did_something);
	
	return true; // no contradiction found
}

void SlinkerGrid::FollowPossibilities(const std::vector<TRule> &solving_rules, vector<SlinkerGrid> &solutions,
									bool guessing_allowed,unsigned int max_n_wanted_solutions)
{
	vector<int*> changed;

	// first apply whatever local rules we can
	bool grid_ok = ApplyRules(solving_rules,changed);

	if(grid_ok) // (otherwise the grid had contradictions => no solutions down this path)
	{
		if(IsFinished())
		{
			if(IsLegal() && find(solutions.begin(),solutions.end(),*this)==solutions.end())
			{
				// we have found a solution not previously found, add it to the list
				solutions.push_back(*this);
			}
		}
		else if(solutions.size()<max_n_wanted_solutions && guessing_allowed) 
			// if there are multiple valid solutions then just finding N is enough 
			// (also guessing (exploration of possibilities) must be allowed)
		{
			int x,y,onoff;
			// follow on/off possibilities for the first unassigned border we find
			bool found_one=false;
			for(x=0;(x<2*X+1 && !found_one);x++)
			{
				for(y=0;(y<2*Y+1 && !found_one);y++)
				{
					if(!IsBorder(x,y) || !IsOnGrid(x,y) || cells[x][y]!=UNKNOWN) 
						continue;
					found_one=true;
					for(onoff=0;onoff<=1;onoff++)
					{
						// a special consideration for speedup: can reject this move 
						// if it would make a loop and not a solution
						if(onoff==1 && WouldMakeALoop(x,y))
						{
							// would this move make a solution?
							SlinkerGrid test(*this);
							test.cells[x][y]=1;
							test.MarkUnknownBordersAsOff();

							if(!test.IsLegal()) // this function is slow but doing this means we can avoid a potentially enormous search avenue
								continue;
							// (if it *is* legal then we will recurse down onto this option and it will be logged as a solution)
						}
						cells[x][y]=onoff;
						FollowPossibilities(solving_rules,solutions,guessing_allowed,max_n_wanted_solutions);
					}
					// revert the test change
					cells[x][y] = UNKNOWN;
				}
			}
		}
	}

	// revert the changes made by applying the rules
	UndoChanges(changed);
}

void SlinkerGrid::UndoChanges(const vector<int*> &changed)
{
	for(vector<int*>::const_iterator it=changed.begin();it!=changed.end();it++)
		**it = UNKNOWN;
}

bool SlinkerGrid::WouldMakeALoop(int x,int y) const
{
	vector<pair<int,int> > dots;
	// add the left/above dot of this border as a starting point
	dots.push_back(IsHorizontalBorder(x,y)?make_pair(x-1,y):make_pair(x,y-1));
	CollectJoinedDots(dots);
	// if the right/below dot is already connected (N.B. border is currently off) then we have a loop
	return(find(dots.begin(),dots.end(),
		IsHorizontalBorder(x,y)?make_pair(x+1,y):make_pair(x,y+1))!=dots.end());
}

bool SlinkerGrid::HasLoop()
{
	// collect connected border into sets
	// if turning any border off leaves its dots connected then return true
	bool found_a_loop=false;
	int x,y;
	for(x=0;x<2*X+1 && !found_a_loop;x++)
	{
		for(y=0;y<2*Y+1 && !found_a_loop;y++)
		{
			if(IsBorder(x,y) && cells[x][y]==1)
			{
				// disable the border temporarily	
				cells[x][y]=0;
				// are the end dots still connected?
				if(WouldMakeALoop(x,y))
					found_a_loop = true;
				// re-enable the border
				cells[x][y]=1;
			}
		}
	}
	return found_a_loop;
}

std::vector<SlinkerGrid> SlinkerGrid::FindSolutions(const vector<TRule> &rules,bool guessing_allowed,int max_n_wanted_solutions) const
{
	// find some solutions
	vector<SlinkerGrid> solutions;
	SlinkerGrid g(*this);
	g.FollowPossibilities(rules,solutions,guessing_allowed,max_n_wanted_solutions);
	return solutions;
}

bool SlinkerGrid::IsConsistent(const std::vector<TRule> &solving_rules,const int& SEARCH_BORDER,int depth)
{
	// first apply the local solving rules - if this grid is inconsistent then we're done
	vector<int*> changed;
	bool is_cons = ApplyRules(solving_rules,changed);

	// if the grid contains any loop at all then this is forbidden (remember this function only used with rule candidates, not whole puzzles)
	if(is_cons && HasLoop())
		is_cons = false;
	
	// otherwise, try a border and recurse down as necessary
	if(is_cons && depth>0)
	{
		int x,y;
		// if both border states give an inconsistent grid (for any border), then the current grid is inconsistent
		bool test_is_cons;
		int n_inconsistent,on_off;
		for(x=SEARCH_BORDER;x<2*X+1-SEARCH_BORDER && is_cons;x++) // for every non-edge entry (don't want off-grid effects here, see FindNewRules)
		{
			for(y=SEARCH_BORDER;y<2*Y+1-SEARCH_BORDER && is_cons;y++)
			{
				int &entry = cells[x][y];
				if(IsBorder(x,y) && entry==UNKNOWN)
				{
					n_inconsistent=0;
					for(on_off=0;on_off<=1;on_off++)
					{
						entry=on_off;
						// with this change, is the grid still consistent, or now in a contradictory state?
						test_is_cons = IsConsistent(solving_rules,SEARCH_BORDER,depth-1); // keep track of recursion depth
						entry=UNKNOWN; // undo our test change
						if(!test_is_cons)
							n_inconsistent++;
						else 
							break; // don't need to check the other possibility: the current grid is consistent
					}
					if(n_inconsistent==2)
						is_cons = false; // the current grid state is inconsistent! (this is a good thing - we have found a rule)
				}
			}
		}
	}
	
	UndoChanges(changed);
	
	return is_cons;
}

SlinkerGrid SlinkerGrid::ReadFromLoopyFormat(const string &s)
{
	istringstream iss(s.c_str());
	char c1,c2;
	int X,Y;
	iss >> X >> c1 >> Y >> c2;
	//if(c1!='x' || c2!=':') throw(runtime_error("Error reading file - expecting the Loopy format."));
	SlinkerGrid g(X,Y);
	int x=0,y=0;
	char c;
	while(iss.good())
	{
		iss >> c;
		if(c>='0' &&c<='3' && x>=0 && x<X && y>=0 && y<Y)
			g.cellValue(x++,y) = (c-'0');
		else if(c>='a' && c<='z')
			x += (c-'a'+1);
		//else throw(runtime_error("Error reading file - unexpected character."));
		while(x>=X)
		{
			y++;
			x-=X;
		}
	}
	return g;
}

