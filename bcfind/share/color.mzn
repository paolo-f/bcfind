
% number of centers
int: n;

% set of centers
set of int: V = 0..n-1;


int: num_edges;
array[1..num_edges, 1..2] of V: E;

% number of colors (hues)
int: nc = 256;
%% int: nc = 32;

% colors
%% array[V] of var 0..nc-1: colorof;
array[V] of var {0,25,40,60,80,100,140,160,180,200,220,240,260,280,300,330}: colorof;


%% var int: obj = sum(i in 1..num_edges) ( min(nc-abs((colorof[E[i,1]] - colorof[E[i,2]]) mod nc), abs((colorof[E[i,1]] - colorof[E[i,2]]) mod nc)) ) ;
%% solve maximize obj;
%% solve :: seq_search([ int_search(colorof, smallest, indomain_median, complete), int_search([obj], first_fail, indomain_median, complete)]) maximize obj;

%% solve :: int_search(colorof, first_fail, indomain_random, complete) satisfy;
solve :: int_search(colorof, first_fail, indomain_random, complete) satisfy;
%% solve satisfy;

constraint
  % there must be no loops 
  forall(i in 1..num_edges) (
    E[i,1] != E[i,2]
  )
  /\
  %% forall(i in 1..num_edges) ( min(nc-abs((colorof[E[i,1]] - colorof[E[i,2]]) mod nc), abs((colorof[E[i,1]] - colorof[E[i,2]]) mod nc)) > 32)
  %% forall(i in 1..num_edges) ( min(nc-abs((colorof[E[i,1]]-colorof[E[i,2]]) mod nc), abs((colorof[E[i,1]]-colorof[E[i,2]]) mod nc)) >= 4);
  %% forall(i in 1..num_edges) ( abs(colorof[E[i,1]]-colorof[E[i,2]])>45 /\ abs(colorof[E[i,1]]-colorof[E[i,2]]) < 315 );
  %% forall(i in 1..num_edges) ( abs(colorof[E[i,1]]-colorof[E[i,2]])>45 /\ abs(colorof[E[i,1]]-colorof[E[i,2]]) < 315 );
  forall(i in 1..num_edges) ( colorof[E[i,1]] != colorof[E[i,2]] );


output 
%% [
%%   "obj: ", show(obj),"\n",
%%   "colorof: ", show(colorof), "\n",
%% ] ++
%% [
%%  show(E[i,1]) ++ "-" ++ show(E[i,2]) ++ " " ++ show(colorof[E[i,1]]) ++ " " ++ show(colorof[E[i,2]]) ++ "\n"
%%  %% ++ show( min(nc-abs((colorof[E[i,1]] - colorof[E[i,2]]) mod nc), abs((colorof[E[i,1]] - colorof[E[i,2]]) mod nc)) ) ++ "\n"
%%   | i in 1..num_edges
%% ] ++ ["\n"]
%% ++
[
 %% "c\t" ++ show(i) ++ "\t" ++ show(colorof[i]*8) ++ "\n" | i in V
 "c\t" ++ show(i) ++ "\t" ++ show(round(fix(255.0*int2float(colorof[i])/360.0))) ++ "\n" | i in V
]
;

%
% data
% 

% The optimal solution is 4 

%% n = 11;
%% num_edges = 20;

%% E = array2d(1..num_edges, 1..2, [
%%  1, 2,
%%  1, 4,
%%  1, 7,
%%  1, 9,
%%  2, 3,
%%  2, 6,
%%  2, 8,
%%  3, 5,
%%  3, 7,
%%  3, 10,
%%  4, 5,
%%  4, 6,
%%  4, 10,
%%  5, 8,
%%  5, 9,
%%  6, 11,
%%  7, 11,
%%  8, 11,
%%  9, 11,
%%  10, 11
%% ]);
