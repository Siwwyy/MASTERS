UE_4.26.2Source\Engine\Binaries\Win64\UE4Editor-Cmd.exe %1 -game -DX12 -rdgimmediate -RenderOffScreen -stdout -windowed -nosound -noailogging -noverifygc -novsync -benchmark -fps=30 -deterministic -ExecCmds="r.SetRes 3840x2160f, r.DepthOfFieldQuality 0, r.DOF.Gather.AccumulatorQuality 0, r.DOF.Gather.PostfilterMethod 0, r.DOF.Gather.EnableBokehSettings 0, r.DOF.Gather.RingCount 0, r.DOF.Scatter.ForegroundCompositing 0, r.DOF.Scatter.BackgroundCompositing 0, r.DOF.Scatter.EnableBokehSettings 0, r.DOF.Recombine.Quality 0, r.DOF.Recombine.EnableBokehSettings 0, r.DOF.TemporalAAQuality 0, r.DOF.Kernel.MaxForegroundRadius 0, r.DOF.Kernel.MaxBackgroundRadius 0, r.MotionBlurQuality 0, r.BloomQuality 0, r.SSR 0, r.SceneColorFringeQuality 0, r.SSR.Quality 0, sg.AntiAliasingQuality 2, r.TemporalAASamples 32, r.MipMapLODBias -2, r.AllowOcclusionQueries 0, r.TextureStreaming 0, r.DumpPass.Allow 1, r.DumpPass.DumpTAABuffer 1"