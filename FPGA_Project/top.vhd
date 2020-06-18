library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library work;
use work.types.all;
entity top is

	port 
	(
		CLK				: in	STD_LOGIC ;
		--VGA--
		HSYNC_VGA		:	OUT	STD_LOGIC;	--horiztonal sync pulse
		VSYNC_VGA		:	OUT	STD_LOGIC;	--vertical sync pulse
		red				:	OUT	STD_LOGIC_VECTOR(0 DOWNTO 0);
		green				:	OUT	STD_LOGIC_VECTOR(0 DOWNTO 0);
		blue				:	OUT	STD_LOGIC_VECTOR(0 DOWNTO 0);
		RESET_VGA		: in	STD_LOGIC ;
		--CAM--
		CAM_SIOC			:	OUT	STD_LOGIC;
		CAM_SIOD			:	INOUT	STD_LOGIC;
		CAM_VSYNC			:	IN		STD_LOGIC;
		CAM_HREF				:	IN		STD_LOGIC;
		CAM_PCLK				:	IN		STD_LOGIC;
		CAM_XCLK				:	OUT	STD_LOGIC;
		CAM_DATA				:	IN		STD_LOGIC_VECTOR(7 DOWNTO 0);
		CAM_RESET			:	OUT 	STD_LOGIC:='1';
		CAM_PWDN				:	OUT	STD_LOGIC:='0';

		CAM_EN			:	in std_LOGIC:='1';		
		edgeDetEn			: 	IN 	std_LOGIC:='1';
		selectInput			: 	IN 	std_LOGIC:='1';
		findOutput		:	in std_logic:='1';
		ledOut		:	OUT	std_LOGIC_VECTOR(3 downto 0):="1111"
		);
end entity;

architecture rtl of top is
component readImage
	port 
	(
		CAM_RESET			:	OUT 	STD_LOGIC:='1';
		CAM_PWDN				:	OUT	STD_LOGIC:='0';
		CAM_VSYNC	:	IN		STD_LOGIC;
		CAM_PCLK		:	IN		STD_LOGIC;
		CAM_DATA		:	IN		STD_LOGIC_VECTOR(7 DOWNTO 0);
		CAM_HREF		:	IN		STD_LOGIC;
		CAM_EN		: 	IN		STD_LOGIC;
		address		: 	out 	std_LOGIC_VECTOR(16 DOWNTO 0);
		CAM_WREN		:	out 	std_LOGIC;
		done			:	out 	std_logic;
		IMG_DATA		: 	OUT 	STD_LOGIC_VECTOR(0 DOWNTO 0)
	);
	end component;
	
component ann 
port 
	(
		clk : in std_logic;
		output : out std_LOGIC;
		reset : in std_LOGIC;
		frame : in img;
		done 	: out std_logic
	);
end component;
component ov7670_driver 
  Port ( 
		iclk50   : in    STD_LOGIC;
		config_finished : out std_logic;
		sioc  : out   STD_LOGIC;
		siod  : inout STD_LOGIC;
		sw : in std_logic_vector( 9 downto 0);
		key : in std_logic_vector( 2 downto 0)
       );
end component;
component vga_controller 
	PORT(
		pixel_clk	:	IN		STD_LOGIC;	--pixel clock at frequency of VGA mode being used
		reset_n		:	IN		STD_LOGIC;	--active low asycnchronous reset
		h_sync		:	OUT	STD_LOGIC;	--horiztonal sync pulse
		v_sync		:	OUT	STD_LOGIC;	--vertical sync pulse
		disp_ena		:	OUT	STD_LOGIC;	--display enable ('1' = display time, '0' = blanking time)
		column		:	OUT	INTEGER;		--horizontal pixel coordinate
		row			:	OUT	INTEGER;		--vertical pixel coordinate
		n_blank		:	OUT	STD_LOGIC;	--direct blacking output to DAC
		n_sync		:	OUT	STD_LOGIC); --sync-on-green output to DAC
END component;
component hw_image_generator 
  PORT(
    ready	 :	 IN 	std_logic;
	 annout	 :	 in std_LOGIC;
	 dataIn	 :	 IN 	std_logic_vector(0 downto 0);
	 address	 :	 out	STD_LOGIC_VECTOR(16 DOWNTO 0);
	 clk		 :  in std_logic;
	 disp_ena :	IN	STD_LOGIC;
    row      :  IN   INTEGER;    --row pixel coordinate
    column   :  IN   INTEGER;    --column pixel coordinate
	 screenEnable: IN std_logic;
    red      :  OUT  STD_LOGIC_VECTOR(0 DOWNTO 0) := (OTHERS => '0');  --red magnitude output to DAC
    green    :  OUT  STD_LOGIC_VECTOR(0 DOWNTO 0) := (OTHERS => '0');  --green magnitude output to DAC
    blue     :  OUT  STD_LOGIC_VECTOR(0 DOWNTO 0) := (OTHERS => '0')); --blue magnitude output to DAC
END component;
component imageRAM
	PORT
	(
		address		: IN STD_LOGIC_VECTOR (16 DOWNTO 0);
		clock		: IN STD_LOGIC  := '1';
		data		: IN STD_LOGIC_VECTOR (0 DOWNTO 0);
		wren		: IN STD_LOGIC ;
		q		: OUT STD_LOGIC_VECTOR (0 DOWNTO 0)
	);
end component;
component edgeDetection

	port 
	(
		pixelIn	   				: in std_logic_vector  (0 downto 0);
		pixelOut 					: out std_logic_vector  (0 downto 0);
		edgeDetEn 	   			: in std_logic;
		address			  			: out std_logic_vector  (16 downto 0);
		edgeDetWE					: out std_logic:='0';
		EdgeDetRD					: out std_logic:='0';
		clk							: in std_logic;
		done							: out std_logic
	);
END component;
component cutImage
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
end component;
component VGAPLL 
	PORT
	(
		inclk0		: IN STD_LOGIC  := '0';
		c0		: OUT STD_LOGIC 
	);
END component;
signal CLKVGA		:STD_LOGIC;
signal CLKCAM		:STD_LOGIC:='0';
signal RAMclk		:std_logic;
signal row	 		:integer;
signal column 		:integer;
signal disp_ena 	:STD_LOGIC;
signal CAM_OUT : std_logic_vector(0 dowNTO 0 );
signal RAMin : std_logic_vector(0 dowNTO 0 );
signal edgeDetPixelOut : std_logic_vector(0 dowNTO 0 );
signal RAMout : std_logic_vector(0 dowNTO 0 );
signal cutPixelOut : std_logic_vector(0 dowNTO 0 );


signal RAMaddress 			: std_logic_VECTOR(16 DOWNTO 0);
signal CAM_ADDRESS 			: std_logic_VECTOR(16 DOWNTO 0);
signal ScreenAddressOut 	: std_logic_VECTOR(16 DOWNTO 0);
signal edgeDetAddressOut 	: std_logic_VECTOR(16 DOWNTO 0);
signal cutAddressOut			: std_logic_VECTOR(16 DOWNTO 0);


signal CAM_WREN 	: std_logic;
signal edgeDetWE 	: std_logic:='0';
signal edgeDetRD  : std_logic:='0';
signal RAMwren 	: std_logic;
signal cutWE		: std_logic;
signal cutRD		: std_logic;

signal frame		: img;
signal annOUT		:std_LOGIC;


--auto
signal findOutputDoneLatch	:	std_LOGIC:='0';
signal findOutputDone	:	std_LOGIC:='0';
signal edgeDetEnAuto		:	std_LOGIC:='1';
signal edgeDetDone		:	std_LOGIC:='0';
signal edgeDetDoneLatch :	std_LOGIC:='0';
signal cutDone				:	std_logic:='0';
signal cutDoneLatch		:	std_logic:='0';
signal selectInputAuto	:	std_LOGIC:='1';
signal findOutputAuto	:	std_LOGIC:='1';
signal CAM_EN_auto		:	std_logic:='0';
signal CAMdone				:	std_logic:='0';
signal CAMdoneLatch		:	std_logic:='0';
signal ScreenPixel		:	std_logic_vector(0 downto 0):="0";
signal ScreenOn			:	std_logic:='0';
--
begin
CAM_XCLK<=CLKCAM;
LEDOut(0)<=ANNOUT;
LEDOut(1)<=CAMdone;
LEDOut(2)<=edgeDetDone;
LEDOut(3)<=findOutputDone;

anncomp: ann port map(
clk => clk,
output => annOUT,
reset => findOutput and findOutputAuto,
frame => frame,
done	=> findOutputDone
);
-------------------------------------------
edge: edgeDetection port map(
pixelIn				=>RAMout,
pixelOut				=>edgeDetPixelOut,
edgeDetEn			=>edgeDetEn and edgeDetEnAuto,
address				=>edgeDetAddressOut,
edgeDetWE			=>edgeDetWE,
clk					=>clk,
edgeDetRD			=>edgeDetRD,
done					=>edgeDetDone
);
--------------------------------------------
cut2828: cutImage port map(
selectInput			=>selectInputAuto,
pixelIn				=>RAMout,
pixelOut				=>cutPixelOut,
address				=>cutAddressOut,
clk					=>clk,
cutWE					=>cutWE,
cutRD					=>cutRD,
frame					=>frame,
cutDone				=>cutDone
);

VGAPLL_inst : VGAPLL PORT MAP (
inclk0	 => CLK,
c0	 => CLKVGA
);
vga : vga_controller PORT MAP (
pixel_clk=>CLKVGA,
reset_n=>RESET_VGA,
h_sync=>HSYNC_VGA,
v_sync=>VSYNC_VGA,
disp_ena=>disp_ena,
column=>column,
row=>row
);

img : hw_image_generator PORT MAP(
ready=>not CAM_WREN,
annout=>annOUT,
dataIn=>RAMout,--ScreenPixel,
address=>ScreenAddressOut,
screenEnable=>ScreenOn,
red=>red,
green=>green,
blue=>blue,
clk=>clk,
row=>row,
column=>column,
disp_ena=>disp_ena
);
driver : ov7670_driver  Port MAP (
iclk50=>CLK,
--config_finished=>cam_en,
sioc=>CAM_SIOC,
siod=>CAM_SIOD,
sw=>"1000000000",
key=>"000"
);
readImg : readImage port  map(
CAM_RESET=>CAM_RESET,
CAM_PWDN=>CAM_PWDN,
CAM_VSYNC=>CAM_VSYNC,
CAM_HREF=>CAM_HREF,
CAM_PCLK=>CAM_PCLK,
CAM_DATA=>CAM_DATA,
CAM_EN=> CAM_EN_auto,
address=>CAM_ADDRESS,
IMG_DATA=>CAM_OUT,
done		=> CAMdone,
CAM_WREN=>CAM_WREN
); 
imageRAM_inst : imageRAM PORT MAP (
		address	=> RAMaddress,
		clock	 	=> clk,
		data	 	=> RAMin,
		wren	 	=> RAMwren,
		q	 		=> RAMout
	);
	
cam_clock : process(clk)
begin
	if rising_edge(clk) then
		clkCAM<=not clkCAM;
	end if;
end process;
RAMaddressdata:process(CAM_WREN,edgeDetWE,edgeDetRD,cutRD,cutWE)
begin
	if CAM_WREN='1' then
		RAMaddress	<=CAM_ADDRESS;
		RAMin			<=CAM_OUT;
	elsif edgeDetWE='1' then
		RAMaddress	<=edgeDetAddressOut;
		RAMin			<=edgeDetPixelOut;
	elsif edgeDetRD ='1' then
		RAMaddress	<=edgeDetAddressOut;
	elsif cutWE='1' then
		RAMaddress	<=cutAddressOut;
		RAMin			<=cutPixelOut;
	elsif cutRD='1' then
		RAMaddress<=cutAddressOut;
	elsif ScreenOn='1' then
		RAMaddress<=ScreenAddressOut;
--		RAMclk<=CLKVGA;
	end if;
end process;


ScreenOn<=  not RAMwren or(CAM_EN_auto and not findOutputAuto); 

RAMwren<=CAM_WREN or edgeDetWE or cutWE;
CAM_EN_auto<=(CAMdoneLatch or (CAM_EN_auto and (not findOutputDoneLatch))) ;
edgeDetEnAuto<= not CAM_EN_auto;
selectInputAuto<=	not edgeDetDoneLatch;
findOutputAuto<=	not cutDoneLatch;

edgeDetDoneLatch<=edgeDetDone;
cutDoneLatch<=cutDone;
findOutputDoneLatch<=findOutputDone;
CAMdoneLatch<=CAMdone;
--auto:process(clk,CAM_EN,edgeDetDone,cutDone,CAMdone,findOutputDone,edgeDetDoneLatch,cutDoneLatch,findOutputDoneLatch,CAMdoneLatch)
----variable i : integer range 0 to 3 := 0;
--begin
--
--if selectInput='0' then
--
--elsif selectInput ='1' then
--	CAM_EN_auto<='1';
--	edgeDetEnAuto<='1';
--	selectInputAuto<=	not edgeDetDoneLatch;
--	edgeDetDoneLatch<=edgeDetDone;
--	findOutputAuto<='1';
--end if;
--end process;
end rtl;
