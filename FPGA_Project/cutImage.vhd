library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library work;
use work.types.all;
entity cutImage is
port
(
	selectInput					: in  std_logic;
	pixelIn						: in  std_logic_vector( 0 downto 0);
	pixelOut						: out std_logic_vector( 0 downto 0);
	address						: out std_logic_vector(16 downto 0);
	clk							: in  std_logic;
	cutWE							: out std_logic:='0';
	cutRD							: out std_logic:='0';
	frame							: out img;
	cutDone						: out std_logic
	);
end entity;

architecture rtl of cutImage is
signal addressTemp : std_logic_vector(16 downto 0):=(others=>'0');
signal break		 : std_logic:='0';
signal state		 : std_logic_vector(1 downto 0):="00";
signal outPixel		:	std_logic_vector(0 downto 0);
signal wren		:	std_logic:='0';

begin
cutDone<=break;
process(clk,selectInput)
variable whitepixel : integer range 0 to 63:=0;
variable r : integer range 0 to 7:=0;
variable c : integer range 0 to 7:=0;
variable rw : integer range 0 to 27:=0;
variable cw : integer range 0 to 27:=0;
variable rt : integer range 0 to 216:=0;
variable ct : integer range 0 to 216:=0;



begin
address<=addressTemp;
pixelOut<=outPixel;
cutWE<=wren;
if selectInput='1' then 
	addressTemp<=(others=>'0');
	wren<='0';
	cutRD<='0';
	r:=0;
	c:=0;
	rw:=0;
	cw:=0;
	rt:=0;
	ct:=0;
	break<='0';
	state<="00";
elsif rising_edge(clk) and break='0' then
	case state is
		when "00"=>--set address for read pixel
			addressTemp<=std_logic_vector(to_unsigned((r+8+rt)*320+(c+48+ct),addressTemp'length));
			cutRD<='1';
			state<="01";
		when "01"=>--check whether pixel is white in 8r8 area
			state<="00";
			if pixelIn="1" then
				whitepixel:=whitepixel+1;
			end if;
			if c<7 then
				c:=c+1;
			else
				c:=0;
				if r<7 then
					r:=r+1;
				else
					state<="10";
					cutRD<='0';
				end if;
			end if;
		when "10"=>--write scaled frame to RAM
			addressTemp<=std_logic_vector(to_unsigned(76800+rw*28+cw,addressTemp'length));--76800
			if whitepixel>5 then 
				outPixel<="1";
			else
				outPixel<="0";
			end if;
			frame(rw*28+cw)<=outPixel(0);
			whitepixel:=0;
			wren<='1';
			state<="11";
		when "11"=>
			wren<='0';
			r:=0;
			c:=0;
			if cw<27 then
				cw:=cw+1;
				ct:=ct+8;
			elsif rw<27 then
				cw:=0;
				ct:=0;
				rw:=rw+1;
				rt:=rt+8;
			else
				break<='1';
				cutRD<='0';
				wren<='0';
			end if;
			state<="00";
	end case;
end if;

end process;
end rtl;