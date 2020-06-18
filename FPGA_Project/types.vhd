library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package types is
	type img	is array(0 to 783) of std_logic;--signal test : img;
	type weightType	is array(0 to 1569) of std_logic_VECTOR(14 downto 0);--signal weight : weightType;
	type layerarray	is array(1 downto 0) of std_logic_vector(17 downto 0);	
	function sigmoid18 (X : std_logic_vector)return std_logic_vector;
	function sum1815 (x : std_logic_vector;y : std_logic_vector)return std_logic_vector;
	function sum1515 (x : std_logic_vector;y : std_logic_vector)return std_logic_vector;
	function mulIw1 (	I : std_logic;w1 : std_logic_vector)return std_logic_vector;
	function mullw2 (l1 : std_logic_vector;w2 : std_logic_vector)return std_logic_vector;
--	function sumofall (test: Img; weight: weightType;i :integer)return layerarray;
end types;

package body types is
function sigmoid18 (X : std_logic_vector)
                 return std_logic_vector is  
  variable temp : std_logic_vecTOR(17 dowNTO 0);
  variable tem2 : std_logic_vecTOR(7 dowNTO 0);

begin
if X(16 downto 10) /= "0000000" then --Xl>="000100" then-- lower -4 and higher 4 -->0 & 1
	temp:=(8=>not X(17),others=>'0' );
else --  (-4 4) -> [0 1]
	if X(17)='1' then--negative
				--s   --xl--  xr:+4  div8
		--return '0'&"000000000"  & '0'&(X(9 downto 8) xor "11")&(std_logic_vector(unsigned(X(7  downto 0) xor "11111111")+1)(7 downto 3));
		temp(17 downto 7):=(others=>'0');
		temp(6 downto 5) :=X(9 downto 8) xor "11";
		tem2:=std_logic_vector(to_unsigned(to_integer(unsigned(X(7  downto 0) xor "11111111"))+1,8));
		temp(4 downto 0) :=tem2(7 downto 3);

	else--positive
				--s   --xl--  xr:+4  div8
		--return '0'&"000000000"  & '1'&(X(9 downto 8) xor "00")&X(7 downto 3);
		temp(17 downto 8):=(others=>'0');
		temp(7):='1';
		temp(6 downto 0) :=X(9 downto 3);

	end if;
end if;
return temp;
end sigmoid18;
function sum1815 (	x : std_logic_vector;--18
						y : std_logic_vector)--15
                 return std_logic_vector is--18
variable xt : unsigned(16 downto 0);
variable yt : unsigned(13 downto 0);

begin
xt:=unsigned(x(16 downto 0));
yt:=unsigned(y(13 dowNTO 0));

if ((x(17) xor y(14))='0') then
	--sum
	return (x(17) & (std_logic_vector(xt+yt)));
else
	--abs(X)>abs(y)
	if xt>yt then	
		return (x(17) & (	std_logic_vector(xt-yt)	));
	elsif xt=yt then 
		return "000000000000000000";
	else
		return (y(14) & (std_logic_vector(yt-xt)));
	end if;
end if;
end sum1815;
function sum1515 (	x : std_logic_vector;--15
						y : std_logic_vector)--15
                 return std_logic_vector is--18
variable xt : unsigned(13 downto 0);
variable yt : unsigned(13 downto 0);
variable re : std_logic_vector(17 downto 0);
begin
xt:=unsigned(x(13 downto 0));
yt:=unsigned(y(13 dowNTO 0));
re:=(others=>'0');
if ((x(14) xor y(14))='0') then --  (++ or --)
	--sum
	re(17):=x(14);
	re(14 downto 0):=std_logic_vector(to_unsigned(to_integer(xt)+to_integer(yt),15));
else									  --	(-+ or +-)
	--abs(X)>abs(y)
	if xt>yt then	
		re(17):=x(14);
		re(13 downto 0):=std_logic_vector(xt-yt);
	elsif xt=yt then 
		re:=(others=>'0');
	else
		re(17):=y(14);
		re(13 downto 0):=std_logic_vector(yt-xt);
	end if;
end if;
return re;
end sum1515;
function mulIw1 (	I : std_logic;
						w1 : std_logic_vector)
                 return std_logic_vector is
variable temp : std_logic_vecTOR(14 dowNTO 0);
begin				--14 13 12 11 10 9  8  7  6  5  4  3  2  1  0
	temp:= w1 and (I& I& I& I& I& I& I& I& I& I& I& I& I& I& I); 
	return temp;
end mulIw1;
function mullw2 (l1 : std_logic_vector;--18 bit (first 9 bit = 0 > cause: sigmoid)
						w2 : std_logic_vector)--15 bit
                 return std_logic_vector is--15 bit
variable o : std_logic_vecTOR(14 downto 0);
variable t : std_logic_vecTOR(22 downto 0);

begin		
	t:=std_logic_vector(unsigned(l1(8 downto 0))*unsigned(w2(13 downto 0)));
	o(14):= w2(14);
	o(13 downto 0):= t(21 downto 8);
	return o;
end mullw2;
--function sumofall (test: Img; weight: weightType;i :integer)
--return layerarray is
--variable flayer : layerarray;
--begin
--	if i<784 then
--		flayer(0):=sum1815(x=>sumofall(test=>test,weight=>weight,i=>i+1)(0),y=>mulIw1(I=>test(i),w1=>weight(i*2)));
--		flayer(1):=sum1815(x=>sumofall(test=>test,weight=>weight,i=>i+1)(1),y=>mulIw1(I=>test(i),w1=>weight(i*2+1)));
--	elsif i=784 then
--		flayer:= ((others=> (others=>'0')));
--	end if;
--return flayer;
--end sumofall;
end types;

