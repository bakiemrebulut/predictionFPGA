library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity readImage is

		port 
	(
		CAM_RESET	:	OUT 	STD_LOGIC:='1';
		CAM_PWDN		:	OUT	STD_LOGIC:='0';
		CAM_VSYNC	:	IN		STD_LOGIC;
		CAM_PCLK		:	IN		STD_LOGIC;
		CAM_DATA		:	IN		STD_LOGIC_VECTOR(7 DOWNTO 0);
		CAM_HREF		:	IN		STD_LOGIC;
		CAM_EN		:  IN 	STD_LOGIC;
		address		: 	out 	std_LOGIC_VECTOR(16 DOWNTO 0);
		CAM_WREN		:	out 	std_LOGIC;
		done 			:	out 	std_logic;
		IMG_DATA		: 	OUT 	STD_LOGIC_VECTOR(0 downto 0)
	);

end entity;

architecture rtl of readImage is



signal pixelDataBuffer      : std_logic_vector(15 downto 0) := (others => '0');
signal addressS      : STD_LOGIC_VECTOR(16 downto 0) := (others => '0');
signal weRAM       : std_logic := '0';
signal latched_vsync : STD_LOGIC := '0';
signal latched_href  : STD_LOGIC := '0';
signal latched_d     : STD_LOGIC_VECTOR (7 downto 0) := (others => '0');
signal dout : std_LOGIC_VECTOR(0 downto 0);
signal CAM_ENLatch : STD_LOGIC;
signal doneSignal : std_logic;
begin
done<=doneSignal;
address <= addressS(16 downto 0);
CAM_WREN <= weRAM;
IMG_DATA<=dout;
blackwhite:process(pixelDataBuffer)
variable sumpx : unsigned(6 downto 0);
begin
sumpx:=unsigned("00"&pixelDataBuffer(15 downto 11))+unsigned("00"&pixelDataBuffer(10 downto 6))+unsigned("00"&pixelDataBuffer(4 downto 0));
if sumpx<47 then 
	dout<="0";
else
	dout<="1";
end if;
end process;
-- This is a bit tricky href starts a pixel transfer that takes 3 cycles
					--        Input   | state after clock tick   
					--         href   | wr_hold    pixelDataBuffer           dout  we address  address_next
					-- cycle -1  x    |    xx      xxxxx xxxxxx xxxxx  xxxxxxxxxxxx  x   xxxx     xxxx
					-- cycle 0   1    |    x1      xxxxx xxxRRR RRGGG  xxxxxxxxxxxx  x   xxxx     addr
					-- cycle 1   0    |    10      RRRRR GGGGGG BBBBB  xxxxxxxxxxxx  x   addr     addr
					-- cycle 2   x    |    0x      GGGBB BBBxxx xxxxx  RRRRGGGGBBBB  1   addr     addr+1
					--										 54321 098765 43210	
stop: process(CAM_VSYNC,CAM_EN)
begin
if CAM_EN='1' then
	doneSignal<='0';
elsif to_integer(unsigned(addressS))=76799 then
	doneSignal<='1'; 
end if;
end process;
capture_process: process(CAM_PCLK,CAM_EN)
variable cycle: std_logic :='0';
   begin 
		if CAM_EN='1' then --reset
			CAM_RESET<='1';
			CAM_PWDN <='0';
			addressS <= (others => '0');
			weRAM		<='0';
			cycle:='0';
		elsif doneSignal='1' then
			CAM_RESET<='0';
			CAM_PWDN <='1';
			addressS <= (others => '0');
			weRAM		<='0';
			cycle:='0';
		elsif falling_edge(CAM_PCLK) then
--			doneSignal<='1';
			
			if latched_href='1' then
				pixelDataBuffer <= pixelDataBuffer( 7 downto 0) & latched_d;
			end if;
			if CAM_VSYNC = '1' then
				addressS     <= (others => '0');
			end if;			
			if cycle='1' then
				weRAM<='1';	
				addressS <= std_logic_vector(unsigned(addressS)+1);
			else 
				weRAM<='0';		
			end if;	
	
		elsif rising_edge(CAM_PCLK) then
			latched_d     <= CAM_DATA;
			latched_href  <= CAM_HREF;
			latched_vsync <= CAM_VSYNC;


			if latched_href='0' then
				cycle:='0';
			else
				cycle:=not cycle;
			end if;
		end if;--reset
      
   end process;


end rtl;

